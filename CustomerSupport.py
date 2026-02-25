from dotenv import load_dotenv
load_dotenv()

import os
from google import genai
from typing_extensions import TypedDict
from operator import add
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
import pickle
from pathlib import Path
from langgraph.types import interrupt, Command


# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
class FileSystemSaver(InMemorySaver):
    """A simple filesystem-backed saver built on top of InMemorySaver.

    It persists the internal storage, writes and blobs to a pickle file after
    every `put`, `put_writes` and `delete_thread` call. This provides simple
    durability across process restarts.
    """

    def __init__(self, path: str | os.PathLike = "checkpoints.pkl", **kwargs) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        # Try to load existing state
        if self.path.exists():
            try:
                with self.path.open("rb") as f:
                    data = pickle.load(f)
                # restore internals if present
                self.storage = data.get("storage", self.storage)
                self.writes = data.get("writes", self.writes)
                self.blobs = data.get("blobs", self.blobs)
            except Exception:
                # If loading fails, continue with empty in-memory structures
                pass

    def _persist(self) -> None:
        try:
            with self.path.open("wb") as f:
                pickle.dump({"storage": self.storage, "writes": self.writes, "blobs": self.blobs}, f)
        except Exception:
            # Best-effort persistence; do not crash the workflow if persistence fails
            pass

    def put(self, config, checkpoint, metadata, new_versions):
        ret = super().put(config, checkpoint, metadata, new_versions)
        self._persist()
        return ret

    def put_writes(self, config, writes, task_id: str, task_path: str = ""):
        super().put_writes(config, writes, task_id, task_path)
        self._persist()

    def delete_thread(self, thread_id: str) -> None:
        super().delete_thread(thread_id)
        self._persist()


checkpointer = FileSystemSaver()
llm = OllamaLLM(
    model="qwen3:8b",   # or llama3:8b
    temperature=0.2,
    num_predict=300
)

def call_llm(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return "ERROR"


# print(call_llm("What is the capital of France?"))
def update_status(current: str, new: str) -> str:
    # This ensures the newest status always replaces the old one
    return new if new else current
class SupportState(TypedDict):
    messages: Annotated[list[str], add]
    user_id: str
    intent: str
    user_profile: dict
    billing_history: dict
    decision: str
    resolution_status: Annotated[str, update_status]

def intentClassifier(state: SupportState)->SupportState:
    print("\n Classifying intent")
    prompt = ChatPromptTemplate.from_template("Identify the user's intent based on the following messages: {messages}. "
    "Respond with ONLY the intent category. Do not explain your reasoning.")
    formatted_prompt = prompt.format(messages="\n".join(state["messages"]))

    intent = call_llm(formatted_prompt)
    print(f"Intent : {intent}")
    return {"intent" : intent}
def fetchUserProfile(state: SupportState)->SupportState:
    print("\n Fetching user profile")
    # Simulate fetching user profile from a database
    user_profile = {
        "name": "John Doe",
        "account_type": "Premium",
        "join_date": "2020-01-15"
    }
    print(f"User Profile : {user_profile}")
    return {"user_profile" : user_profile}
def fetchBillingHistory(state: SupportState)->SupportState:
    print("\n Fetching billing history")
    # Simulate fetching billing history from a database
    billing_history = {
        "last_payment_date": "2024-05-01",
        "last_payment_amount": "$49.99",
        "outstanding_balance": "$49.99"
    }
    print(f"Billing History : {billing_history}")
    return {"billing_history" : billing_history}
# def mergeInformation(state: SupportState)->SupportState:
#     print("\n Merging information for decision making")
#     # Simulate merging information to make a decision
#     intent = state["intent"].strip().lower()
#     if 'billing' in intent and state["billing_history"]["outstanding_balance"] == "$-49.99":
#         decision = "Offer refund"
#     else:
#         decision = "Escalate to human agent"
#     print(f"Decision : {decision}")
#     return {"decision" : decision}

def mergeInformation(state: SupportState)->SupportState:
    print("\n Merging information for decision making")
    # Simulate merging information to make a decision
    prompt = ChatPromptTemplate.from_template("Fetch the information from billing_history: {billing_history} and user_profile : {user_profile} and based on the intent: {intent} and the message : {messages} provided decide whether to 'Offer refund' or 'Escalate to human agent'. "
                                              "if you moret than 50% confident then offer refund else escalate to human agent. if the balance is negative and intent is related to billing then it should be more likely to offer refund."
    "Respond with either this 'Offer refund' or 'Escalate to human agent' . Do not explain your reasoning.")
    prompt_4 = prompt.format(billing_history=state["billing_history"], user_profile=state["user_profile"], intent=state["intent"], messages="\n".join(state["messages"]))
    decision = call_llm(prompt_4)
    print(f"Decision : {decision}")
    return {"decision" : decision}
def offerRefund(state: SupportState)->SupportState:
    print("\n Offering refund to the customer")
    # Simulate offering a refund
    resolution_status = "Refund offered to the customer"
    print(f"Resolution Status : {resolution_status}")
    return {"resolution_status" : resolution_status}
# def escalateToHumanAgent(state: SupportState)->SupportState:
#     print("\n Escalating to human agent")
#     # Simulate escalating to a human agent
#     resolution_status = "Escalated to human agent for further assistance"
#     print(f"Resolution Status : {resolution_status}")
#     return {"resolution_status" : resolution_status}
# return {"resolution_status": "PENDING_APPROVAL"}
def escalateToHumanAgent(state: SupportState):
    print("\nEscalating to human agent...")

    # Pause execution and wait for resume payload
    is_approved = interrupt({
        "question": "Do you want to proceed with refund? (true/false)",
        "case_details": state
    })

    # Validate resume payload (important for safety)
    if not isinstance(is_approved, bool):
        raise ValueError("Human approval must be boolean (True/False)")

    if is_approved:
        print("--- Human Approved ---")
        return Command(
            update={"decision": "accepted"}, 
            goto="finalResponse"
        )
    else:
        print("--- Human Rejected ---")
        return Command(
            update={"decision": "rejected"}, 
            goto="finalResponse"
        )

def finalResponse(state: SupportState)->SupportState:
    print("\n Generating final response for the customer")
    decision = state["decision"].strip().lower()


    if "accepted" in decision:
        final_response = "Refund approved by human agent."
    elif "rejected" in decision:
        final_response = "Refund rejected by human agent."
    elif "offer" in decision:
        final_response = "Refund processed automatically."
    else:
        final_response = "Escalated to support."
    return {"resolution_status" : final_response}
def fn(state: SupportState)-> str:
    print("Running function to decide next step")
    if state["decision"] == "Offer refund":
        return "offer_refund"
    else:
        return "Escalate_human_agent"

workflow = StateGraph(SupportState)
workflow.add_node("intentClassifier", intentClassifier)
workflow.add_node("fetchUserProfile", fetchUserProfile)
workflow.add_node("fetchBillingHistory", fetchBillingHistory)
workflow.add_node("mergeInformation", mergeInformation)
workflow.add_node("offerRefund", offerRefund)
workflow.add_node("escalateToHumanAgent", escalateToHumanAgent)
workflow.add_node("finalResponse", finalResponse)
# workflow.add_node("fetchUserProfile", fetchUserProfile)

workflow.add_edge(START, "intentClassifier")
workflow.add_edge("intentClassifier", "fetchUserProfile")
workflow.add_edge("intentClassifier", "fetchBillingHistory")    
workflow.add_edge("fetchBillingHistory", "mergeInformation")    
workflow.add_edge("fetchUserProfile", "mergeInformation")    
# workflow.add_edge("mergeInformation", "mergeInformation")   
workflow.add_conditional_edges("mergeInformation", fn,{"offer_refund":"offerRefund", "Escalate_human_agent":"escalateToHumanAgent"})
#  
workflow.add_edge("offerRefund", "finalResponse")    
workflow.add_edge("escalateToHumanAgent", "finalResponse")    
workflow.add_edge("finalResponse", END) 
graph = workflow.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "support_1"}}
result = graph.invoke({
    "messages": ["I was doing the paytment but no acknowledgement received so i did again and amount is deducted twice from my account. I want a refund for extra payment."],
    "user_id": "User_112",
    "intent": "",
    "user_profile": {},
    "billing_history": {},
    "decision": "",
    "resolution_status": ""
}, config=config)

# Check if human review is needed
# if result.get("resolution_status") == "PENDING_APPROVAL":
#     print("\n--- HUMAN REVIEW REQUIRED ---")
#     print("Message: Refund request requires human approval")
#     print("\nCurrent Case State:")
#     print(result)

#     decision = input("\nEnter decision (accept/reject): ").strip().lower()

#     if decision == "accept":
#         result["resolution_status"] = "ACCEPTED - Refund approved by human agent"
#     else:
#         result["resolution_status"] = "REJECTED - Refund denied by human agent"
state = graph.get_state(config)

if state.next: # If there is a 'next' node, it means we are paused at an interrupt
    print("\n--- ⏸️ HUMAN REVIEW REQUIRED ---")
    
    # Access the interrupt payload correctly
    # state.tasks[0].interrupts contains the data passed to interrupt()
    current_interrupt = state.tasks[0].interrupts[0]
    print(f"Question: {current_interrupt.value['question']}")
    
    # Get user input
    user_input = input("Approve? (yes/no): ").strip().lower()
    approved = True if user_input == "yes" else False

    # Resume the graph
    print("\n--- ▶️ Resuming Workflow ---")
    result = graph.invoke(Command(resume=approved), config=config)

    # Check the decision here
    print(f"DEBUG: Decision in final_output: {result.get('decision')}")
else:
    print("Workflow finished without interruption.")
print("\nFinal Result:")
print(result)