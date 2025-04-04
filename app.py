from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv

# Step 1: Set OpenAI API Key
try:
    load_dotenv()
except Exception as e:
    print(f"Error loading environment variables: {e}")

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set in the environment variables.")
if not os.getenv("TAVILY_API_KEY"):
    raise EnvironmentError("TAVILY_API_KEY is not set in the environment variables.")

# Define search tool

search = TavilySearchResults(max_results=1)

# Define the tool for context retrieval
@tool
def retrieve_context(query: str):
    """Search for relevant documents."""
    # Example URL configuration
    print('tool called')
    urls = [
        "https://docs.python.org/3/tutorial/index.html",
        "https://realpython.com/python-basics/",
        "https://www.learnpython.org/"
    ]
    # Load documents
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs)
    
    # Create VectorStore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="python_docs",
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])

tools = [retrieve_context,search]
tool_node = ToolNode(tools)

# OpenAI LLM model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools(tools)

# Function to decide whether to continue or stop the workflow
def should_continue(state: MessagesState) -> Literal["tools", END]:
    print('should_continue')
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, go to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, finish the workflow
    return END

# Function that invokes the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}  # Returns as a list to add to the state

# Define the workflow with LangGraph
workflow = StateGraph(MessagesState)

# Add nodes to the graph
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Connect nodes
workflow.add_edge(START, "agent")  # Initial entry
workflow.add_conditional_edges("agent", should_continue)  # Decision after the "agent" node
workflow.add_edge("tools", "agent")  # Cycle between tools and agent

# Configure memory to persist the state
checkpointer = MemorySaver()

# Compile the graph into a LangChain Runnable application
app = workflow.compile(checkpointer=checkpointer)

# Execute the workflow
final_state = app.invoke(
    {"messages": [HumanMessage(content="What is the weather in Los Angeles?")]},
    config={"configurable": {"thread_id": 42}}
)

# Show the final response
print(final_state["messages"][-1].content)