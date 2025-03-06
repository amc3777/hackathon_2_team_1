from typing_extensions import Literal
from pydantic import BaseModel
import random
import re
import argparse

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langchain_google_vertexai import ChatVertexAI
# from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool

members = ["habits", "summaries", "alerts"]
# options = members + ["FINISH"]

# Define the router schema with Pydantic BaseModel
class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""
    # next: Literal[*options]
    next: Literal[*members]
    # next: str

# Define graph state
class State(MessagesState):
    next: str

# Define the nodes
def router(state: State) -> Command[Literal[*members]]:
    # print("outer loop: router")

    llm = ChatVertexAI(model="gemini-2.0-flash-lite-001", temperature=0)

    router_prompt = (
               """
               <instructions>
               \nYou are the critical routing component of a healthcare AI system.
               \nYour response will only be one of 3 options: 'FINISH', 'habits', 'summaries', or 'alerts'.
               \nBased on <user query>, decide if the user is...
               \n(1) requesting custom workout routines and diet plans -> respond 'habits'.
               \n(2) OR requesting a simplified summary of complex medical information -> respond 'summaries'.
               \n(3) OR need to be alerted because of health risks based on <patient data>  -> respond 'alerts'.
               \nRemember, only output 'habits', 'summaries', or 'alerts'.
               \n</instructions>
               """)


    messages = [
        {"role": "system", "content": router_prompt},] + state.get("messages", [])
    # print(messages)
    resp = llm.with_structured_output(schema=Router).invoke(messages)
    # print(resp)
    value = resp.next
    messages = state.get("messages", []) + [AIMessage(content=value)]

    goto = value
    # if value == "FINISH":
    #     goto = END

    # note how Command allows you to BOTH update the graph state AND route to the next node
    return Command(
        # this is the state update
        update={"messages": messages, "next": value},
        # this is a replacement for an edge
        goto=goto,
    )


def habits(state: State) -> Command[Literal["__end__"]]:
    # print("outer loop: habits")

    # llm = ChatVertexAI(model="gemini-2.0-flash")
    llm = ChatVertexAI(model="gemini-2.0-pro-exp-02-05", temperature=0)

    habits_prompt = (
               """
               <instructions>
               \nYou are a healthy habits expert. You are a critical component of a healthcare AI system.
               \nYou have been empowered by a healthcare provider to guide patients towards better health outcomes.
               \nGenerate customized workout routines and meal plans based on <patient data>.
               \nBe friendly and respectful.
               \nBe concise and structured.
               \n</instructions>
               """)

    messages = [
        {"role": "system", "content": habits_prompt},] + state.get("messages", [])

    # resp = llm.invoke(messages, tools=[VertexTool(google_search={})],)
    resp = llm.invoke(messages)
    # print(resp)
    messages = state.get("messages", []) + [AIMessage(content=resp.content)]

    return Command(
        # this is the state update
        update={"messages": messages, "next": "FINISH"},
        # this is a replacement for an edge
        goto=END,
    )

def summaries(state: State) -> Command[Literal["__end__"]]:
    # print("outer loop: summaries")

    # llm = ChatVertexAI(model="gemini-2.0-flash")
    llm = ChatVertexAI(model="gemini-2.0-pro-exp-02-05", temperature=0)

    summaries_prompt = (
               """
               <instructions>
               \nYou are an expert at simplifying complicated written medical content. You are a critical component of a healthcare AI system.
               \nYou have been empowered by a healthcare provider to guide patients towards better health outcomes.
               \nSummarize complex medical information (e.g., from doctor's notes, research papers) in <user query> into layman's terms.
               \nAnswer patient questions in <user query> about their conditions and treatments.
               \n<patient data> will also be provided, in case it is needed.
               \nBe friendly and respectful.
               \nBe concise and structured.
               \n</instructions>
               """)

    messages = [
        {"role": "system", "content": summaries_prompt},] + state.get("messages", [])

    # resp = llm.invoke(messages, tools=[VertexTool(google_search={})],)
    resp = llm.invoke(messages)
    # print(resp)
    messages = state.get("messages", []) + [AIMessage(content=resp.content)]

    return Command(
        # this is the state update
        update={"messages": messages, "next": "FINISH"},
        # this is a replacement for an edge
        goto=END,
    )

def alerts(state: State) -> Command[Literal["__end__"]]:
    # print("outer loop: alerts")

    # llm = ChatVertexAI(model="gemini-2.0-flash")
    llm = ChatVertexAI(model="gemini-2.0-pro-exp-02-05", temperature=0)

    alerts_prompt = (
               """
               <instructions>
               \nYou are a medical risk-assessment expert. You are a critical component of a healthcare AI system.
               \nYou have been empowered by a healthcare provider to guide patients towards better health outcomes.
               \nAnalyze <patient data> and proactively identify potential health risks or deviations from normal measurment levels.
               \nGenerate alerts and offer personalized recommendations for intervention.
               \nBe friendly and respectful.
               \nBe concise and structured.
               \n</instructions>
               """)

    messages = [
        {"role": "system", "content": alerts_prompt},] + state.get("messages", [])

    # resp = llm.invoke(messages, tools=[VertexTool(google_search={})],)
    resp = llm.invoke(messages)
    # print(resp)
    messages = state.get("messages", []) + [AIMessage(content=resp.content)]

    return Command(
        # this is the state update
        update={"messages": messages, "next": "FINISH"},
        # this is a replacement for an edge
        goto=END,
    )

builder = StateGraph(State)
builder.add_edge(START, "router")
builder.add_node(router)
builder.add_node(habits)
builder.add_node(summaries)
builder.add_node(alerts)
# note: there are no edges between nodes A, B and C!

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


def process_messages(messages):
    for message in messages:
        if isinstance(message, AIMessage):
            if message.content not in ('habits', 'summaries', 'alerts'):
                if isinstance(message.content, list):
                    all_parts = ' '.join(map(lambda part: str(part).strip(), message.content))
                    print(f"AI: {all_parts}") 
                else:
                    print(f"AI: {message.content}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healthcare AI Application") # Create argument parser
    parser.add_argument("--user_query", type=str, required=True, help="The user's query/input text") # Add argument for user query
    parser.add_argument("--patient_data", type=str, default="", help="Optional patient data (can be empty string)") # Optional patient data
    parser.add_argument("--thread_id", type=int, required=True, help="thread ID") # thread ID for memory
    args = parser.parse_args() # Parse the arguments from command line

    user_input_text = args.user_query # Get user query from arguments
    patient_data_text = args.patient_data # Get patient data from arguments
    thread_id = args.thread_id # Get thread ID from arguments

    full_user_input = f"<user_query>{user_input_text}</user_query> <patient_data>{patient_data_text}</patient_data>"
    message = {"messages": [HumanMessage(content=full_user_input)]}
    config = {"configurable": {"thread_id": thread_id}}

    print(f"Human: {user_input_text}") # print human input to console

    for chunk in graph.stream(message, config, stream_mode="updates"):
        for key, value in chunk.items():
            messages = value['messages']
            process_messages(messages)