from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from loguru import logger

from agents.chart import ChartAgent
from agents.planner import PlannerAgent
from agents.researcher import ResearchAgent
from lib.workflow.graph import StateGraph

# Load environment variables
load_dotenv()

if __name__ == '__main__':
    planner = PlannerAgent()
    researcher = ResearchAgent()
    charter = ChartAgent()

    workflow = StateGraph(MessagesState)

    workflow.add_agent(planner, is_started=True)
    workflow.add_agent(researcher)
    workflow.add_agent(charter)

    graph = workflow.compile()

    msg = "Draw chart compare area of UK, US and China in 2024"
    events = graph.stream(
        {"messages": HumanMessage(msg)},
        {"recursion_limit": 150},
    )

    for s in events:
        logger.info(f"event: {s=}")
