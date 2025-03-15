from langgraph.constants import START
from langgraph.graph import StateGraph as BaseStateGraph
from loguru import logger

from lib.agent.agent import Agent


class StateGraph(BaseStateGraph):
    def __init__(self, t):
        super().__init__(t)

    def add_agent(self, a: Agent, is_started=False):
        logger.info(f"Add agent {a.name} to graph")
        self.add_node(a.name, a.get_note)
        if is_started:
            self.set_start(a)

    def set_start(self, a: Agent):
        logger.info(f"Set start agent to {a.name}")
        self.add_edge(START, a.name)
