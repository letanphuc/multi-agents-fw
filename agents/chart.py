from lib.agent.agent import Agent
from lib.tools.repl import python_repl_tool


class ChartAgent(Agent):
    name = "chart_generator"
    tools = [python_repl_tool]
    prompt = ("You are agent that can only generate charts. "
              "Store your chart image in `./out` directory. "
              "When the file was created, you complete the task.")
