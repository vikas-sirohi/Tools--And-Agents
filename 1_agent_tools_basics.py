from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model = "mixtral-8x7b-32768",
        temperature = 0,
        max_tokens = 100,
        max_retries = 2,
        )

# Our First Custom Tool
def get_current_time(*args, **kwargs):
    """Returns current date time in Hrs:Mins PM/AM"""
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

tools = [
    Tool(
        name = "Time",
        func = get_current_time,
        description = "Useful for when you need to know the current time.",
    ),
]

# Prompt that our agent will follow.
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm = llm,
    tools = tools,
    prompt = prompt,
    stop_sequence = True,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tools,
    verbose = True,
)

response = agent_executor.invoke({"input":"what time is it?"})
print("response: ", response)
