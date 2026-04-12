# Leap0.dev sandbox tools for CrewAI

Run shell commands and Python in an isolated [Leap0](https://leap0.dev) cloud sandbox instead of local Docker.

## Install

```bash
pip install "crewai-tools[leap0]"
```

Set `LEAP0_API_KEY` in your environment (see Leap0 docs).

## Import

```python
from crewai_tools.tools.leap0_sandbox_tool import create_leap0_sandbox_toolkit
```

You can also import from the package barrel: `from crewai_tools import create_leap0_sandbox_toolkit` (with `[leap0]` installed).

## Usage

Create a `Leap0Client`, create a sandbox, build the toolkit, pass the tools to an agent, run the crew, then clean up.

```python
import os

from crewai import Agent, Crew, Task
from leap0 import Leap0Client

from crewai_tools.tools.leap0_sandbox_tool import create_leap0_sandbox_toolkit


def main() -> None:
    if not os.environ.get("LEAP0_API_KEY"):
        raise SystemExit("Set LEAP0_API_KEY")

    client = Leap0Client()
    sandbox = client.sandboxes.create()
    toolkit, code_tools = create_leap0_sandbox_toolkit(client, sandbox)

    try:
        developer = Agent(
            role="Python Developer",
            goal="Solve tasks with code in the sandbox",
            backstory="You run code via leap0_execute_code and related tools.",
            tools=code_tools,
            verbose=True,
        )
        task = Task(
            description="Compute factorial of 6 with Python and print the result.",
            agent=developer,
        )
        crew = Crew(agents=[developer], tasks=[task])
        crew.kickoff()
    finally:
        toolkit.cleanup()


if __name__ == "__main__":
    main()
```

## Tools

| Tool | Purpose |
|------|---------|
| `leap0_execute_code` | Run Python (optional `libraries_used` → `pip install` first). Code is written under `/tmp` and executed with `python3`. |
| `leap0_execute_command` | Run a shell command; result includes `exit_code` and output. |
| `leap0_read_file` | Read a UTF-8 file (absolute path only). |
| `leap0_write_file` | Write a UTF-8 text file (absolute path only). |

## Cleanup

`toolkit.cleanup()` deletes the sandbox (if `delete_sandbox_on_cleanup=True`) and closes the client (if `close_client_on_cleanup=True`). Use a `try` / `finally` so cleanup runs after `crew.kickoff()` even on errors.

If multiple toolkits share one `Leap0Client`, set `close_client_on_cleanup=False` on all but the last toolkit, or manage `client.close()` yourself.

## Sandbox reference

Pass the object returned by `client.sandboxes.create()` (recommended). If you pass a sandbox id string, the toolkit calls `client.sandboxes.get(id)` once at construction to obtain a handle with `process` and `filesystem` clients (requires network at that moment).
