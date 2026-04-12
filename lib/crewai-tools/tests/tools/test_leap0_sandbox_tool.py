"""Tests for Leap0 sandbox CrewAI toolkit."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("leap0")

from crewai_tools.tools.leap0_sandbox_tool import (
    Leap0SandboxToolkit,
    create_leap0_sandbox_toolkit,
)


def _make_toolkit() -> tuple[Leap0SandboxToolkit, MagicMock, object]:
    client = MagicMock()
    process = MagicMock()
    filesystem = MagicMock()
    sandbox = SimpleNamespace(id="sb-test", process=process, filesystem=filesystem)
    toolkit = Leap0SandboxToolkit(
        client,
        sandbox,
        delete_sandbox_on_cleanup=False,
        close_client_on_cleanup=False,
    )
    return toolkit, client, sandbox


def test_sandbox_id_from_namespace() -> None:
    toolkit, _, _ = _make_toolkit()
    assert toolkit.sandbox_id == "sb-test"


def test_execute_shell_success() -> None:
    toolkit, _, sandbox = _make_toolkit()
    sandbox.process.execute.return_value = SimpleNamespace(
        stdout="ok\n", stderr="", exit_code=0
    )
    out = toolkit.execute_shell("echo ok")
    assert "exit_code: 0" in out
    assert "ok" in out
    sandbox.process.execute.assert_called_once()
    assert sandbox.process.execute.call_args.kwargs["command"] == "echo ok"


def test_execute_shell_leap0_error() -> None:
    from leap0 import Leap0Error

    toolkit, _, sandbox = _make_toolkit()
    sandbox.process.execute.side_effect = Leap0Error("boom", 500)
    out = toolkit.execute_shell("false")
    assert "Leap0 API error" in out
    assert "boom" in out


def test_execute_python_pip_failure() -> None:
    toolkit, _, sandbox = _make_toolkit()
    sandbox.process.execute.return_value = SimpleNamespace(
        stdout="nope", stderr="", exit_code=1
    )
    out = toolkit.execute_python("print(1)", ["badpkg"])
    assert "pip install failed" in out
    assert "badpkg" in out


def test_execute_python_runs_file() -> None:
    toolkit, _, sandbox = _make_toolkit()

    def exec_side_effect(*_args: object, **kwargs: object) -> SimpleNamespace:
        cmd = kwargs.get("command", "")
        if "pip" in cmd:
            return SimpleNamespace(stdout="", stderr="", exit_code=0)
        if "crewai_leap0_" in cmd and cmd.strip().startswith("python3"):
            return SimpleNamespace(stdout="hello\n", stderr="", exit_code=0)
        return SimpleNamespace(stdout="", stderr="", exit_code=0)

    sandbox.process.execute.side_effect = exec_side_effect
    out = toolkit.execute_python('print("hello")', [])
    assert "exit_code: 0" in out
    sandbox.filesystem.write_bytes.assert_called_once()
    wb = sandbox.filesystem.write_bytes.call_args.kwargs
    assert wb["path"].startswith("/tmp/crewai_leap0_")
    assert b"print" in wb["content"]
    sandbox.filesystem.delete.assert_called_once_with(path=wb["path"])


def test_execute_python_deletes_script_when_run_raises() -> None:
    from leap0 import Leap0Error

    toolkit, _, sandbox = _make_toolkit()

    def exec_side_effect(*_args: object, **kwargs: object) -> SimpleNamespace:
        cmd = kwargs.get("command", "")
        if "pip" in cmd:
            return SimpleNamespace(stdout="", stderr="", exit_code=0)
        if "crewai_leap0_" in cmd and cmd.strip().startswith("python3"):
            raise Leap0Error("run failed", 500)
        return SimpleNamespace(stdout="", stderr="", exit_code=0)

    sandbox.process.execute.side_effect = exec_side_effect
    out = toolkit.execute_python("print(1)", [])
    assert "Leap0 API error" in out
    sandbox.filesystem.write_bytes.assert_called_once()
    sandbox.filesystem.delete.assert_called_once()


def test_read_file_invalid_path() -> None:
    toolkit, _, _ = _make_toolkit()
    assert "absolute" in toolkit.read_sandbox_file("relative.txt").lower()


def test_write_file_invalid_path() -> None:
    toolkit, _, _ = _make_toolkit()
    assert "absolute" in toolkit.write_sandbox_file("rel", "x").lower()


def test_cleanup_deletes_and_closes() -> None:
    client = MagicMock()
    sandbox = SimpleNamespace(
        id="sb-1",
        process=MagicMock(),
        filesystem=MagicMock(),
    )
    toolkit = Leap0SandboxToolkit(
        client,
        sandbox,
        delete_sandbox_on_cleanup=True,
        close_client_on_cleanup=True,
    )
    toolkit.cleanup()
    client.sandboxes.delete.assert_called_once_with(sandbox)
    client.close.assert_called_once()


def test_cleanup_idempotent() -> None:
    toolkit, client, _ = _make_toolkit()
    toolkit._delete_sandbox_on_cleanup = True
    toolkit._close_client_on_cleanup = True
    toolkit.cleanup()
    toolkit.cleanup()
    assert client.sandboxes.delete.call_count == 1
    assert client.close.call_count == 1


def test_string_sandbox_id_resolves_via_get() -> None:
    client = MagicMock()
    resolved = SimpleNamespace(
        id="resolved-id",
        process=MagicMock(),
        filesystem=MagicMock(),
    )
    client.sandboxes.get.return_value = resolved
    toolkit = Leap0SandboxToolkit(
        client,
        "sandbox-id-str",
        delete_sandbox_on_cleanup=False,
        close_client_on_cleanup=False,
    )
    client.sandboxes.get.assert_called_once_with("sandbox-id-str")
    assert toolkit.client is client
    assert toolkit._sandbox is resolved
    assert toolkit.sandbox_id == "sandbox-id-str"


def test_create_factory_returns_tools() -> None:
    client = MagicMock()
    sandbox = SimpleNamespace(
        id="sb-2",
        process=MagicMock(),
        filesystem=MagicMock(),
    )
    toolkit, tools = create_leap0_sandbox_toolkit(
        client,
        sandbox,
        delete_sandbox_on_cleanup=False,
        close_client_on_cleanup=False,
    )
    names = {t.name for t in tools}
    assert names == {
        "leap0_execute_code",
        "leap0_execute_command",
        "leap0_read_file",
        "leap0_write_file",
    }
    assert toolkit.get_tools() == tools
