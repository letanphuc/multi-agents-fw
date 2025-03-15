from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    PromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.types import Command
from loguru import logger


class Agent:
    name: str = None
    llm: BaseChatModel = ChatOpenAI(model="gpt-4o-mini")
    tools: list = []
    _based_prompt_text = '''
        You are a helpful AI assistant, collaborating with other assistants.
        Use the provided tools to progress towards answering the question.
        If you are unable to fully answer, that's OK, another assistant with different tools
        will help where you left off. Execute what you can to make progress.
        If you or any of the other assistants have the final answer or deliverable,
        place `END` in the `goto` field and your final answer in the `message` field.
    '''

    _output_prompt_text = (
        '''
        Here is list of available agents to handoff goto: {agents}
        RESPONSE IN JSON FORMAT, NO MORE ADDITIONAL TEXT
        {{
            "message": text \\ your text message in plain text, 
                            \\ explain your detailed result, what is next agent should do
            "goto": ... \\ name of the next agent to be called, one of above agents list, 
                        \\ or `END` if you have the final answer
        }}
        '''
    )

    prompt: str = ""
    agents = {}
    verbose = True

    def __init__(self):
        if not self.prompt:
            raise ValueError(f"{self.__class__}: prompt is required")

        self.p = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(self._based_prompt_text),
                SystemMessagePromptTemplate.from_template(self.prompt),
                SystemMessagePromptTemplate.from_template(self._output_prompt_text),
                MessagesPlaceholder(optional=True, variable_name="agent_scratchpad"),
            ],
            input_variables=[
                "agent_scratchpad",
                "chat_history",
                "agents",
                "intermediate_steps",
            ],
        )

        self.agents[self.name] = self

    def invoke(self, state: MessagesState) -> Command:
        logger.debug(f'{self.name}: {state=}')
        msg_count = len(self.p.messages)
        p = ChatPromptTemplate(
            messages=self.p.messages[:msg_count - 1] + state['messages'] + [self.p.messages[msg_count - 1]],
            input_variables=self.p.input_variables
        )
        agent = create_openai_tools_agent(self.llm, self.tools, prompt=p)
        for _ in range(3):
            try:
                agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=self.verbose)
                result = agent_executor.invoke({"agents": self.agents.keys()})
                data = JsonOutputParser().invoke(result['output'])
                messages = state['messages'] + [AIMessage(content=f'{self.name}: ' + data['message'])]
                return Command(update={"messages": messages}, goto=data['goto'])
            except Exception as e:
                logger.error(f"{self.name}: {e}")
        raise Exception(f"{self.name}: failed to invoke")

    def get_note(self, state):
        r = self.invoke(state)
        logger.info(f"{self.name}: {r=}")
        return r
