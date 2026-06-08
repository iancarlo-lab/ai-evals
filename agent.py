import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

# Load environment variables from your .env file
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. DEFINE THE TOOLS
# ==========================================
@tool
def fetch_stock_price(ticker: str) -> float:
    """Fetches the current stock price for a given ticker symbol."""
    mock_market = {"AAPL": 175.0, "MSFT": 420.0, "GOOG": 150.0}
    return mock_market.get(ticker.upper(), 100.0)

@tool
def calculate_valuation_multiplier(price: float, earnings_per_share: float) -> float:
    """Calculates the P/E ratio given the current price and earnings per share."""
    if earnings_per_share <= 0:
        return 0.0
    return round(price / earnings_per_share, 2)

tools = [fetch_stock_price, calculate_valuation_multiplier]
tool_node = ToolNode(tools)

# ==========================================
# 2. DEFINE THE GRAPH STATE
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ==========================================
# 3. DEFINE THE LOGIC & ROUTER
# ==========================================
# We use gpt-4o-mini with temperature=0 for deterministic tool calling
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
).bind_tools(tools)

def call_model(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    # If the LLM didn't request a tool call, we finish the execution loop
    if not last_message.tool_calls:
        return END
    # Otherwise, route to the "tools" node
    return "tools"

# ==========================================
# 4. BUILD AND COMPILE THE GRAPH (The 'app')
# ==========================================
workflow = StateGraph(AgentState)

# Add our processing units (nodes)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set up the conditional wiring (edges)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

# THIS IS THE MISSING PIECE: Compiling it creates the 'app' variable
app = workflow.compile()

# ==========================================
# 5. SANITY CHECK RUNNER
# ==========================================
if __name__ == "__main__":
    print("Running agent test...")
    inputs = {"messages": [HumanMessage(content="What is the price of AAPL?")]}
    
    # Stream updates so you can watch the state evolution live in the terminal
    for output in app.stream(inputs, stream_mode="updates"):
        print("\n--- State Update ---")
        print(output)