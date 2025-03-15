from typing import Annotated

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

repl = PythonREPL()


@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user.
        Do not re-run this tool if it's successful and code's not changed """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    if result:
        return f"Successfully executed, here is stdout: \n{result}\n\n"
    else:
        return f"Successfully executed, the file was created"
