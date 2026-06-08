# LangGraph Stock Agent & Evaluation Demo

This repository demonstrates how to build a stateful stock agent using **LangGraph** (powered by Gemini 2.5 Pro via `ChatGoogleGenerativeAI`) and evaluate its execution trajectory and outputs using **LangSmith**.

---

## Architecture Overview

The agent is modeled as a state machine using LangGraph:

1. **State**: Keeps track of the chat messages sequence.
2. **Nodes**:
   - `agent`: Invokes the LLM (Gemini 2.5 Pro) with tools bound to it.
   - `tools`: Executes Python tools based on tool calls requested by the agent.
3. **Edges**:
   - Entry point: `agent`.
   - Conditional route: `should_continue` inspects the last message. If there are tool calls, routes to `tools`; otherwise, it finishes at `END`.
   - After `tools` execution, loops back to the `agent`.

### Bound Tools
- **`fetch_stock_price`**: Fetches the stock price for a given ticker symbol (mocked: AAPL=$175.0, MSFT=$420.0, GOOG=$150.0).
- **`calculate_valuation_multiplier`**: Calculates the P/E ratio given the current price and earnings per share.

---

## Project Structure

- `agent.py`: Definition of tools, graph state, nodes, and entry point compilation. Running it directly performs a basic query stream check.
- `eval_agent.py`: Sets up a LangSmith test dataset, runs target evaluations, and evaluates execution trajectory accuracy and final math correctness.
- `.gitignore`: Configured to exclude `.env`, virtual environments, and python cache files.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iancarlo-lab/langgraph_demo.git
   cd langgraph_demo
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies:**
   Ensure you have installed the required libraries:
   ```bash
   pip install langgraph langchain-core langchain-google-genai python-dotenv langsmith
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory (this is ignored by Git):
   ```env
   # Google API Key for Gemini
   GEMINI_API_KEY=your_gemini_api_key_here

   # LangSmith Configuration (for evaluation/monitoring)
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=langgraph-stock-evals
   ```

---

## Running the Project

### 1. Run the Agent Locally
Run the `agent.py` script to test a simple natural language query (e.g., asking for the price of AAPL) and watch the LangGraph state transitions:
```bash
python agent.py
```

### 2. Run Evaluations with LangSmith
Run the `eval_agent.py` script to set up a test dataset in LangSmith and evaluate the agent's trajectory and answers:
```bash
python eval_agent.py
```
This script evaluates:
- **Trajectory Sequence Accuracy**: Checks if the correct tools were invoked in the exact expected order.
- **Output Correctness**: Confirms the expected mathematical outputs are present in the final reply.
