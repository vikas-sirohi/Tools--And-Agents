from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_groq import ChatGroq

load_dotenv()

# Creating Tools

def get_current_time(*args, **kwargs):
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

def search_wikipedia(query):
    """Searches wikipedia and returns simmry of the first result."""
    from wikipedia import summary

    try:
        return summary(query, sentences = 2)
    except:
        return "I couldn't find any information on that."
    
tools = [
    Tool(
        name = "Time",
        func = get_current_time,
        description = "Useful for when you nedd to know the current time.",
    ),
    Tool(
        name = "Wikipedia",
        func = search_wikipedia,
        description = "Useful for when you need to know information about a topic.",
    ),
]

prompt = hub.pull("hwchase17/structured-chat-agent")

llm = ChatGroq(model = "mixtral-8x7b-32768",
        temperature = 0,
        max_tokens = 100,
        max_retries = 2,
        )

# ConversationBufferMemory stores the conversation history, 
# allowing the agent to maintain context across interactions

memory = ConversationBufferMemory(
    memory_key = "chat_history", return_messages = True,
)

agent = create_structured_chat_agent(
    llm = llm,
    tools = tools,
    prompt = prompt
)

# AgentExecutor is responsible for managing the interaction 
# between the user input, the agent, and the tools.
agent_executor = AgentExecutor.from_agent_and_tools(
    agent = agent,
    tools = tools,
    verbose = True,
    memory = memory,
    handle_parsing_errors = True,
)

# Setting initial instructions or context
initial_messages = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content = initial_messages))

# Chat loop to intract with the agent.

while True:
    user_input = input("\nUser: ")
    if user_input.lower() == "exit":
        break

    # Add user message to conversation memory
    memory.chat_memory.add_message(HumanMessage(content = user_input))

    # Invoke the agent
    response = agent_executor.invoke({"input": user_input})
    print("\nAI: ", response["output"])

    memory.chat_memory.add_message(AIMessage(content = response["output"]))