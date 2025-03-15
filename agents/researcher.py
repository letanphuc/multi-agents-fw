from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from lib.agent.agent import Agent


class ResearchAgent(Agent):
    name = "researcher"
    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [TavilySearchResults(max_results=5)]
    prompt = "You are researcher agent that can only do research"
