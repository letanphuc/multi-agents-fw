from langchain_openai import ChatOpenAI

from lib.agent.agent import Agent


class PlannerAgent(Agent):
    name = "planner"
    prompt = ("You are agent that can only do planning. "
              "Start by creating list of task to complete user's request.")
