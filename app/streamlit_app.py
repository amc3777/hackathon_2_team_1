from typing_extensions import Literal
from pydantic import BaseModel
import json
import random
# import re
# import argparse
import streamlit as st  # Import Streamlit

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langchain_google_vertexai import ChatVertexAI
# from google.cloud.aiplatform_v1beta1.types import Tool as VertexTool

from google.cloud import bigquery

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


# Modified process_messages for Streamlit
def process_messages(messages):
    for message in messages:
        if isinstance(message, AIMessage):
            if message.content not in ('habits', 'summaries', 'alerts'):
                if isinstance(message.content, list):
                    all_parts = ' '.join(map(lambda part: str(part).strip(), message.content))
                    st.markdown(f"**AI:** {all_parts}")
                else:
                    st.markdown(f"**AI:** {message.content}")

def get_patient_ids_from_bigquery():
    """Fetches patient IDs from BigQuery."""
    client = bigquery.Client()
    query = "SELECT DISTINCT patient_id FROM `andrewcooley-genai-tests.ai_summit.patient_data` ORDER BY patient_id ASC"
    query_job = client.query(query)
    results = query_job.result()
    patient_ids = [row.patient_id for row in results]
    return patient_ids

def get_patient_data_from_bigquery(patient_id):
    """Fetches patient data from BigQuery based on patient ID."""
    client = bigquery.Client()
    query = f"SELECT * FROM `andrewcooley-genai-tests.ai_summit.patient_data` WHERE patient_id = {patient_id}"
    query_job = client.query(query)
    results = query_job.result()
    patient_data_list = [dict(row.items()) for row in results]
    patient_data = json.dumps(patient_data_list, indent=2, default=str)

    return patient_data

# Streamlit UI
st.title("Healthcare AI Assistant")

patient_ids = get_patient_ids_from_bigquery()
selected_patient = st.sidebar.selectbox("Patient ID", patient_ids)

user_query  = st.selectbox(
    'What type of assistance do you need?',
    ('I need a custom workout routine and meal plan.', 'Summarize complex medical information.', 'Analyze my personal health data for risks.')
)
patient_data = get_patient_data_from_bigquery(selected_patient)
st.text_area("Patient Data", value=patient_data, height=400)

if st.button("Get Support"):
        full_user_input = f"<user_query>{user_query}</user_query> <patient_data>{patient_data}</patient_data>"
        message = {"messages": [HumanMessage(content=full_user_input)]}
        config = {"configurable": {"thread_id": random.randint(0, 1000)}}

        st.markdown(f"**Human:** {user_query}")

        for chunk in graph.stream(message, config, stream_mode="updates"):
            for key, value in chunk.items():
                messages = value['messages']
                process_messages(messages)