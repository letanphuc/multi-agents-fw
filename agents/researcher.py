from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from lib.agent.agent import Agent


class ResearchAgent(Agent):
    name = "researcher"
    tools = [TavilySearchResults(max_results=5)]
    prompt = "You are researcher agent that can only do research"
