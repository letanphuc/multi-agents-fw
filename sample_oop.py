from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from loguru import logger

# Load environment variables
load_dotenv()


class Agent:
    def __init__(self, name: str, llm: ChatOpenAI, tools: list, prompt: str):
        self.name = name
        self.agent = create_react_agent(llm, tools, prompt=prompt)

    def invoke(self, state: MessagesState, next_node: str) -> Command[Literal["researcher", "chart_generator", END]]:
        logger.debug(f'{self.name}: {state=}')
        result = self.agent.invoke(state)
        goto = self.get_next_node(result["messages"][-1], next_node)
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name=self.name
        )
        return Command(update={"messages": result["messages"]}, goto=goto)

    @staticmethod
    def get_next_node(last_message: BaseMessage, goto: str):
        if "FINAL ANSWER" in last_message.content:
            return END
        logger.info(f"goto = {goto}")
        return goto


class ResearchAgent(Agent):
    def __init__(self, llm: ChatOpenAI, search_tool):
        super().__init__(
            name="researcher",
            llm=llm,
            tools=[search_tool],
            prompt="You can only do research. You are working with a chart generator colleague."
        )


def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}\n\nIf you have completed all tasks, respond with FINAL ANSWER."


class ChartAgent(Agent):
    def __init__(self, llm: ChatOpenAI, execution_tool):
        super().__init__(
            name="chart_generator",
            llm=llm,
            tools=[execution_tool],
            prompt="You can only generate charts. You are working with a researcher colleague."
        )


if __name__ == '__main__':
    model = ChatOpenAI()
    repl = PythonREPL()
    tavily_tool = TavilySearchResults(max_results=5)

    research_agent = ResearchAgent(model, tavily_tool)
    chart_agent = ChartAgent(model, python_repl_tool)


    def research_node(state: MessagesState) -> Command[Literal["chart_generator", END]]:
        return research_agent.invoke(state, "chart_generator")


    def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
        return chart_agent.invoke(state, "researcher")


    workflow = StateGraph(MessagesState)
    workflow.add_node("researcher", research_node)
    workflow.add_node("chart_generator", chart_node)
    workflow.add_edge(START, "researcher")
    graph = workflow.compile()

    events = graph.stream(
        {"messages": [("user",
                       "First, get the UK's GDP over the past 5 years, then make a line chart of it. Once you make the chart, finish.")]},
        {"recursion_limit": 150},
    )

    for s in events:
        logger.info(f"event: {s=}")
