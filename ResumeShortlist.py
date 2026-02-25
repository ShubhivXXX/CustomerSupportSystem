from dotenv import load_dotenv
load_dotenv()

import os
from google import genai
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
# from google.genai import types
# client = genai.Client(api_key=os.getenv("google_api_key"))
# def call_gemini(prompt: str) -> str:
#     response = client.models.generate_content(
#         model="gemini-flash-latest",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             temperature=0.3,
#             max_output_tokens=500,
#         )
#     )
    
#     return response.text

# Initialize Ollama model
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

class State(TypedDict):
    application:str
    experience_level: str
    skill_match : str
    response : str
    branch : str


workflow = StateGraph(State)
def categorize_experience(state: State)->State:
    print("\n Categorizing experience Level")
    prompt = ChatPromptTemplate.from_template("Identify the experience level: 'Entry-Level', 'Mid-level', or 'Senior-Level'. "
    "Respond with ONLY the category name. Do not explain your reasoning. "
    "Application: {application}")
    formatted_prompt = prompt.format(application = state["application"])
    exp_lvl = call_llm(formatted_prompt)
    print(f"Experience : {exp_lvl}")
    return {"experience_level" : exp_lvl}

def categorize_branch(state: State)->State:
    print("\n finding the degree")
    promt_2 = ChatPromptTemplate.from_template("based on the application tell me what could be the branch of the person it can only be among these 'Computer Science' or 'electronics' or 'electrical' or 'Mechanical ' or 'Electrical and electronics'"
                                               "Respond with ONLY the category name. Do not explain your reasoning. "
                                             "Application:{application}")
    formatted_promt_2 = promt_2.format(application = state["application"])
    branch = call_llm(formatted_promt_2)
    print(branch)
    return {"branch":branch}
def add_skill_match(state: State)->State:
    print("\n finding the skill match")
    promt_3 = ChatPromptTemplate.from_template("based on the application and the experience level and branch tell me 'accepted' or 'rejected' based on skills match for a callibration who have an experience on Yamaha Bikes as well as powertrain role. It should be 1 word string only"
                                               "Respond with ONLY the accepted or rejected. Do not explain your reasoning. "
                                             "Application:{application} Experience Level:{experience_level} Branch:{branch}")
    formatted_promt_3 = promt_3.format(application = state["application"], experience_level=state["experience_level"], branch=state["branch"])
    skill_match = call_llm(formatted_promt_3)
    print(skill_match)
    return {"skill_match":skill_match}
def reject_application(state: State)->State:
    print("\n rejecting the application")

    return {"response":"Skills do not match for the backend role. Rejected the application"}
def schedule_interview(state: State)->State:
    print("\n scheduling the interview")
    return {"response":"Interview Scheduled for tomorrow at 10 am"}
def escalate_to_hr(state: State)->State:
    print("\n escalating to hr")
    return {"response":"escalating_to_hr"}
def fn(state: State)-> str:
    print("Running function to decide next step")
    if state["skill_match"] == "rejected" and state["experience_level"]=="Senior-Level":
        return "escalate_to_hr"
    elif state["skill_match"] == "accepted":
        return "schedule_interview"
    else:
        return "reject_application"
workflow.add_node("categorize_experience", categorize_experience)
workflow.add_node("categorize_branch", categorize_branch)
workflow.add_node("add_skill_match", add_skill_match)
workflow.add_node("schedule_interview", schedule_interview)
workflow.add_node("reject_application", reject_application)
workflow.add_node("escalate_to_hr", escalate_to_hr)



workflow.add_edge(START, "categorize_experience")
workflow.add_edge("categorize_experience", "categorize_branch")
workflow.add_edge("categorize_branch", "add_skill_match")
workflow.add_conditional_edges("add_skill_match", fn ,{
    "schedule_interview": "schedule_interview",
    "reject_application": "reject_application",
    "escalate_to_hr": "escalate_to_hr"
})

workflow.add_edge("escalate_to_hr", END)
workflow.add_edge("reject_application", END)
workflow.add_edge("schedule_interview", END)
graph = workflow.compile()
result = graph.invoke({
    "application": "8 years experience in backend engineering in a product based company. Proficient in Java, Python and cloud technologies. Led a team of 5 engineers and successfully delivered multiple projects on time. Strong problem-solving skills and experience with microservices architecture.",
    "experience_level": "",
    "skill_match": "",
    "response": "",
    "branch":""
})

print(result)
