import os
from dotenv import load_dotenv

# 1. Load configuration and environment
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate, EvaluationResult
from langsmith.schemas import Example, Run
from langchain_core.messages import HumanMessage

# Import the compiled LangGraph workflow from your previous file
from agent import app

# Initialize the LangSmith client
client = Client()

# ==========================================
# 2. CREATE THE EVALUATION DATASET
# ==========================================
DATASET_NAME = "Stock_Agent_Trajectory_Dataset"

def setup_dataset():
    """Creates a test dataset in LangSmith if it doesn't already exist."""
    if not client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME, 
            description="Evaluates tool execution order and final mathematical outputs."
        )
        
        # Test Case 1: Requires sequential tool use (Fetch -> Calculate)
        client.create_example(
            inputs={"query": "Calculate the P/E ratio for AAPL assuming an EPS of 7.0."},
            outputs={
                "expected_trajectory": ["fetch_stock_price", "calculate_valuation_multiplier"],
                "expected_final_contains": "25"  # Mock price 175.0 / 7.0 = 25.0
            },
            dataset_id=dataset.id
        )
        
        # Test Case 2: Only requires a single tool step
        client.create_example(
            inputs={"query": "What is the current stock price of MSFT?"},
            outputs={
                "expected_trajectory": ["fetch_stock_price"],
                "expected_final_contains": "420"  # Mock price for MSFT is 420.0
            },
            dataset_id=dataset.id
        )
        print(f"🎉 Created dataset '{DATASET_NAME}' with 2 evaluation examples.")
    else:
        print(f"✅ Dataset '{DATASET_NAME}' already exists. Proceeding to evaluation...")

# ==========================================
# 3. DEFINE CUSTOM EVALUATORS
# ==========================================
def evaluate_agent_trajectory(run: Run, example: Example) -> EvaluationResult:
    """
    TRAJECTORY EVALUATION: Digs into the execution trace tree to confirm 
    the agent invoked the correct tools in the exact order intended.
    """
    expected_tools = example.outputs.get("expected_trajectory", [])
    actual_tool_calls = []
    
    # LangSmith flattens the execution tree. We check child runs for executed tools.
    if run.child_runs:
        for child in run.child_runs:
            # Look for the LangGraph node responsible for calling tools
            if child.name == "tools" and child.child_runs:
                # Extract the names of the underlying Python functions that ran
                for tool_run in child.child_runs:
                    actual_tool_calls.append(tool_run.name)
                    
    # Strict sequence matching evaluation
    passed = actual_tool_calls == expected_tools
    
    return EvaluationResult(
        key="trajectory_sequence_accuracy",
        score=1.0 if passed else 0.0,
        comment=f"Expected path: {expected_tools} | Actual path taken: {actual_tool_calls}"
    )

def evaluate_final_output(run: Run, example: Example) -> EvaluationResult:
    """
    COMPONENT EVALUATION: Simple code-based check to ensure the mathematical 
    result exists within the final natural language answer.
    """
    # Extract the final output from LangGraph's state dictionary
    messages_state = run.outputs.get("messages", [])
    if not messages_state:
        return EvaluationResult(key="output_correctness", score=0.0, comment="No messages returned.")
        
    # Grab the string representation of the very last message from the agent
    final_reply = str(messages_state[-1])
    expected_str = example.outputs.get("expected_final_contains")
    
    passed = expected_str in final_reply
    return EvaluationResult(
        key="output_correctness",
        score=1.0 if passed else 0.0,
        comment=f"Looked for substring '{expected_str}' inside final agent response."
    )

# ==========================================
# 4. RUN THE EVALUATION EXPERIMENT LOOP
# ==========================================
def target_runner(inputs: dict):
    """Maps dataset inputs to the graph state schema required by LangGraph."""
    # Convert raw text query from the dataset into the HumanMessage graph state
    graph_input = {"messages": [HumanMessage(content=inputs["query"])]}
    return app.invoke(graph_input)

if __name__ == "__main__":
    # Ensure our evaluation dataset is active
    setup_dataset()
    
    print("🚀 Triggering evaluation experiment run against LangSmith...")
    
    # Run the experiment suite
    experiment_results = evaluate(
        target_runner,
        data=DATASET_NAME,
        evaluators=[evaluate_agent_trajectory, evaluate_final_output],
        experiment_prefix="gemini-trajectory-v1"
    )
    
    print("\n✨ Evaluation Complete! Check your LangSmith project dashboard to view the trace tree matrices.")