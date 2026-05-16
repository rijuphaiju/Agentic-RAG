import streamlit as st
import os
import json
import pandas as pd
from rag_pipeline import load_faiss_index as load_s1, rag_query as run_s1
from stage2_rag_pipeline import rag_query_stage2 as run_s2

# Page Config
st.set_page_config(page_title="RAG Research Lab", layout="wide")
st.title("✦ RAG Agentic Research Lab")

# Sidebar - Configuration
with st.sidebar:
    st.header("Pipeline Settings")
    stage_version = st.radio("Pipeline Stage", ["Stage 1: Baseline", "Stage 2: Agentic (Verified)"])
    
    st.divider()
    st.subheader("Model Info")
    st.info("LLM: llama3.2\n\nRetriever: FAISS + all-MiniLM-L6-v2")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load resources (Cached)
@st.cache_resource
def get_pipeline():
    # We use the same index for both as Stage 2 is backward compatible
    return load_s1()

index, embedder, passages = get_pipeline()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            with st.expander("View Source Context & Scores"):
                st.json(message["metadata"])

# Chat Input
if prompt := st.chat_input("Ask a question about the dataset..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            if stage_version == "Stage 1: Baseline":
                st.write("Retrieving and generating...")
                result = run_s1(prompt, index, embedder, passages)
                response = result["answer"]
                metadata = {"retrieved_titles": [p["title"] for p in result["retrieved_passages"]]}
            else:
                st.write("Retrieving context...")
                st.write("Verifying faithfulness...")
                result = run_s2(prompt, index, embedder, passages, verbose=False)
                response = result["answer"]
                metadata = {
                    "faithfulness": result["faithfulness_score"],
                    "confidence": result["confidence_score"],
                    "fallback_used": result["used_fallback"],
                    "retrieved_titles": [p["title"] for p in result["retrieved_passages"]]
                }
            status.update(label="Response generated!", state="complete", expanded=False)

        # Display answer
        st.markdown(response)
        
        # Display Agentic indicators for Stage 2
        if stage_version != "Stage 1: Baseline":
            cols = st.columns(3)
            cols[0].metric("Faithfulness", f"{result['faithfulness_score']:.2f}")
            cols[1].metric("Confidence", f"{result['confidence_score']:.2f}")
            cols[2].metric("Fallback", "Yes" if result["used_fallback"] else "No")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": response, 
        "metadata": metadata
    })