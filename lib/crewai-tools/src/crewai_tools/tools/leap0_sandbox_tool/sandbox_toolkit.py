"""CrewAI tools backed by a Leap0.dev remote sandbox."""

from __future__ import annotations

import asyncio
import logging
import shlex
from typing import Any, Protocol, runtime_checkable
import uuid

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

_LEAP0_INSTALL_MSG = (
    "The leap0 package is required for Leap0 sandbox tools. "
    'Install it with: pip install "crewai-tools[leap0]"'
)

try:
    from leap0 import Leap0Client, Leap0Error
    from leap0.models.sandbox import SandboxRef, sandbox_id_of

    _HAS_LEAP0 = True
except ImportError:
    _HAS_LEAP0 = False

    Leap0Client = Any  # type: ignore[misc,assignment]
    SandboxRef = Any  # type: ignore[misc,assignment]

    class Leap0Error(Exception):  # type: ignore[no-redef]
        """Placeholder when the leap0 optional dependency is not installed."""

    def sandbox_id_of(_obj: Any) -> str:  # type: ignore[no-redef]
        raise ImportError(_LEAP0_INSTALL_MSG)


def _require_leap0() -> None:
    if not _HAS_LEAP0:
        raise ImportError(_LEAP0_INSTALL_MSG)


def _process_stdout_stderr(result: object) -> tuple[int, str]:
    """Build combined text from :class:`leap0.models.process.ProcessResult`-style objects."""
    exit_code = int(getattr(result, "exit_code", 0))
    out = getattr(result, "stdout", None)
    err = getattr(result, "stderr", None)
    out_s = out if isinstance(out, str) else ""
    err_s = err if isinstance(err, str) else ""
    if err_s and out_s:
        return exit_code, f"{out_s}\n--- stderr ---\n{err_s}"
    return exit_code, out_s or err_s


@runtime_checkable
class _SandboxWithPublicId(Protocol):
    """Structural type for sandbox handles with a string ``id`` (SDK models, tests).

    The Leap0 ``SandboxIdentifiable`` protocol is not ``@runtime_checkable``; this
    mirrors it so we can use ``isinstance`` instead of ``hasattr``.
    """

    id: str


class ExecuteCodeInput(BaseModel):
    """Input for running Python in the Leap0 sandbox."""

    code: str = Field(
        ...,
        description=(
            "Python 3 source to execute in the sandbox. Print or log results you "
            "need the agent to see."
        ),
    )
    libraries_used: list[str] = Field(
        default_factory=list,
        description=(
            "PyPI package names to pip install before running (e.g. numpy, pandas). "
            "Use empty list if standard library only."
        ),
    )


class ExecuteCommandInput(BaseModel):
    """Input for a shell command in the Leap0 sandbox."""

    command: str = Field(
        ...,
        description="Shell command to run inside the sandbox (Linux environment).",
    )


class ReadFileInput(BaseModel):
    """Input for reading a file from the sandbox."""

    path: str = Field(
        ...,
        description="Absolute path in the sandbox (must start with /).",
    )


class WriteFileInput(BaseModel):
    """Input for writing a text file in the sandbox."""

    path: str = Field(
        ...,
        description="Absolute path in the sandbox (must start with /).",
    )
    content: str = Field(..., description="UTF-8 text to write.")


class Leap0SandboxToolkit:
    """Holds Leap0 client and sandbox; shared by CrewAI tools.

    Use :func:`create_leap0_sandbox_toolkit` to build a toolkit and tool list.
    Call :meth:`cleanup` when finished (e.g. after ``crew.kickoff()``) to delete
    the sandbox and optionally close the client.
    """

    def __init__(
        self,
        client: Leap0Client,
        sandbox: SandboxRef,
        *,
        default_timeout: int = 30 * 60,
        delete_sandbox_on_cleanup: bool = True,
        close_client_on_cleanup: bool = True,
    ) -> None:
        _require_leap0()
        self._client = client
        if isinstance(sandbox, str):
            self._sandbox = client.sandboxes.get(sandbox)
            self._sandbox_id = sandbox
        else:
            self._sandbox = sandbox
            if isinstance(sandbox, _SandboxWithPublicId):
                self._sandbox_id = str(sandbox.id)
            else:
                self._sandbox_id = sandbox_id_of(sandbox)
        self._default_timeout = default_timeout
        self._delete_sandbox_on_cleanup = delete_sandbox_on_cleanup
        self._close_client_on_cleanup = close_client_on_cleanup
        self._sandbox_deleted = False
        self._client_closed = False
        self.tools: list[BaseTool] = []
        self._setup_tools()

    def _setup_tools(self) -> None:
        self.tools = [
            Leap0ExecuteCodeTool(self),
            Leap0ExecuteCommandTool(self),
            Leap0ReadFileTool(self),
            Leap0WriteFileTool(self),
        ]

    @property
    def client(self) -> Leap0Client:
        return self._client

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    def _execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> tuple[int, str]:
        effective = timeout if timeout is not None else self._default_timeout
        result = self._sandbox.process.execute(
            command=command,
            timeout=effective,
        )
        return _process_stdout_stderr(result)

    def _format_process_result(self, exit_code: int, output: str) -> str:
        return f"exit_code: {exit_code}\n---\n{output}"

    def execute_shell(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> str:
        """Run a shell command and return exit code plus stdout/stderr text."""
        try:
            code, out = self._execute(command, timeout=timeout)
            return self._format_process_result(code, out)
        except Leap0Error as exc:
            return f"Leap0 API error: {exc.message} (status={exc.status_code})"

    def execute_python(
        self,
        code: str,
        libraries_used: list[str],
        *,
        timeout: int | None = None,
    ) -> str:
        """Install optional packages, write code to a temp file, run with python3."""
        try:
            for lib in libraries_used:
                pkg = lib.strip()
                if not pkg:
                    continue
                pip_cmd = f"python3 -m pip install -q {shlex.quote(pkg)}"
                pip_code, pip_out = self._execute(pip_cmd, timeout=timeout)
                if pip_code != 0:
                    return (
                        f"pip install failed for {pkg!r} "
                        f"(exit_code={pip_code})\n---\n{pip_out}"
                    )

            remote_path = f"/tmp/crewai_leap0_{uuid.uuid4().hex}.py"  # noqa: S108
            try:
                self._sandbox.filesystem.write_bytes(
                    path=remote_path,
                    content=code.encode("utf-8"),
                )
                py_cmd = f"python3 {shlex.quote(remote_path)}"
                run_code, run_out = self._execute(py_cmd, timeout=timeout)
                return self._format_process_result(run_code, run_out)
            finally:
                try:
                    self._sandbox.filesystem.delete(path=remote_path)
                except Exception as exc:
                    logger.warning(
                        "Leap0 temp script delete failed (%s): %s",
                        remote_path,
                        exc,
                    )
        except Leap0Error as exc:
            return f"Leap0 API error: {exc.message} (status={exc.status_code})"

    def read_sandbox_file(self, path: str) -> str:
        """Read a file from the sandbox; path must be absolute."""
        if not path.startswith("/"):
            return "Error: path must be absolute (start with /)."
        try:
            data = self._sandbox.filesystem.read_bytes(path=path)
            return data.decode("utf-8", errors="replace")
        except Leap0Error as exc:
            return f"Leap0 API error: {exc.message} (status={exc.status_code})"

    def write_sandbox_file(self, path: str, content: str) -> str:
        """Write UTF-8 text to a path in the sandbox."""
        if not path.startswith("/"):
            return "Error: path must be absolute (start with /)."
        try:
            self._sandbox.filesystem.write_bytes(
                path=path,
                content=content.encode("utf-8"),
            )
            return f"Wrote {len(content.encode('utf-8'))} bytes to {path}."
        except Leap0Error as exc:
            return f"Leap0 API error: {exc.message} (status={exc.status_code})"

    def get_tools(self) -> list[BaseTool]:
        return self.tools

    def cleanup(self) -> None:
        """Delete the sandbox (if configured) and close the client (if configured)."""
        if (
            (not self._delete_sandbox_on_cleanup or self._sandbox_deleted)
            and (not self._close_client_on_cleanup or self._client_closed)
        ):
            return

        if self._delete_sandbox_on_cleanup and self._sandbox is not None:
            sandbox_ref = self._sandbox
            try:
                self._client.sandboxes.delete(sandbox_ref)
                logger.info("Leap0 sandbox %s deleted", self._sandbox_id)
                self._sandbox_deleted = True
                self._sandbox = None
            except Exception as exc:
                logger.warning("Leap0 sandbox delete failed: %s", exc)

        if self._close_client_on_cleanup and not self._client_closed:
            try:
                self._client.close()
                self._client_closed = True
            except Exception as exc:
                logger.warning("Leap0 client close failed: %s", exc)


class Leap0ExecuteCodeTool(BaseTool):
    """Run Python in the Leap0 sandbox."""

    name: str = "leap0_execute_code"
    description: str = (
        "Execute Python 3 code in an isolated Leap0.dev sandbox. "
        "Installs listed PyPI packages first when provided. "
        "Use print() so output is visible in the tool result."
    )
    args_schema: type[BaseModel] = ExecuteCodeInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit: Leap0SandboxToolkit, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.toolkit = toolkit

    def _run(
        self,
        code: str,
        libraries_used: list[str] | None = None,
    ) -> str:
        libs = libraries_used if libraries_used is not None else []
        return self.toolkit.execute_python(code, libs)

    async def _arun(
        self,
        code: str,
        libraries_used: list[str] | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._run,
            code=code,
            libraries_used=libraries_used,
        )


class Leap0ExecuteCommandTool(BaseTool):
    """Run a shell command in the Leap0 sandbox."""

    name: str = "leap0_execute_command"
    description: str = (
        "Run a shell command in the Leap0 sandbox (Linux). "
        "Returns exit_code and combined output."
    )
    args_schema: type[BaseModel] = ExecuteCommandInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit: Leap0SandboxToolkit, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.toolkit = toolkit

    def _run(self, command: str) -> str:
        return self.toolkit.execute_shell(command)

    async def _arun(self, command: str) -> str:
        return await asyncio.to_thread(self._run, command=command)


class Leap0ReadFileTool(BaseTool):
    """Read a file from the Leap0 sandbox."""

    name: str = "leap0_read_file"
    description: str = "Read a text file from the Leap0 sandbox (absolute path only)."
    args_schema: type[BaseModel] = ReadFileInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit: Leap0SandboxToolkit, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.toolkit = toolkit

    def _run(self, path: str) -> str:
        return self.toolkit.read_sandbox_file(path)

    async def _arun(self, path: str) -> str:
        return await asyncio.to_thread(self._run, path=path)


class Leap0WriteFileTool(BaseTool):
    """Write a text file in the Leap0 sandbox."""

    name: str = "leap0_write_file"
    description: str = (
        "Create or overwrite a UTF-8 text file in the Leap0 sandbox "
        "(absolute path only)."
    )
    args_schema: type[BaseModel] = WriteFileInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit: Leap0SandboxToolkit, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.toolkit = toolkit

    def _run(self, path: str, content: str) -> str:
        return self.toolkit.write_sandbox_file(path, content)

    async def _arun(self, path: str, content: str) -> str:
        return await asyncio.to_thread(
            self._run,
            path=path,
            content=content,
        )


def create_leap0_sandbox_toolkit(
    client: Leap0Client,
    sandbox: SandboxRef,
    *,
    default_timeout: int = 30 * 60,
    delete_sandbox_on_cleanup: bool = True,
    close_client_on_cleanup: bool = True,
) -> tuple[Leap0SandboxToolkit, list[BaseTool]]:
    """Build a :class:`Leap0SandboxToolkit` and its CrewAI tools.

    Args:
        client: Authenticated ``Leap0Client`` (e.g. ``Leap0Client()`` with
            ``LEAP0_API_KEY`` set).
        sandbox: Return value of ``client.sandboxes.create()``, or a sandbox id
            string (resolved with ``client.sandboxes.get`` inside the toolkit).
        default_timeout: Default process timeout in seconds.
        delete_sandbox_on_cleanup: If True, :meth:`Leap0SandboxToolkit.cleanup`
            calls ``client.sandboxes.delete(sandbox)``.
        close_client_on_cleanup: If True, :meth:`Leap0SandboxToolkit.cleanup`
            calls ``client.close()``.

    Returns:
        ``(toolkit, tools)`` — pass ``tools`` to ``Agent(..., tools=tools)``.
    """
    _require_leap0()
    toolkit = Leap0SandboxToolkit(
        client,
        sandbox,
        default_timeout=default_timeout,
        delete_sandbox_on_cleanup=delete_sandbox_on_cleanup,
        close_client_on_cleanup=close_client_on_cleanup,
    )
    return toolkit, toolkit.get_tools()
