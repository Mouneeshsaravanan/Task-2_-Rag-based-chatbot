import streamlit as st
from document_loader import load_and_split_documents
from vector_store import create_vectorstore, retrieve_relevant_docs
from llm_handler import get_llm, generate_rag_response, generate_pure_llm_response
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY")
if 'hf_api_key' not in st.session_state:
    st.session_state.hf_api_key = os.getenv("HF_API_KEY")

# Center the title using markdown and CSS
st.markdown(
    """
    <h1 style='text-align: center;'> Techjays </h1>
    """,
    unsafe_allow_html=True
)

# Clear chat history and vector store button
if st.button("Clear Chat History and Data"):
    st.session_state.chat_history = []
    st.session_state.vectorstore = None
    st.success("Chat history and vector store cleared!")

# Upload multiple PDFs
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    docs = load_and_split_documents(uploaded_files)
    st.session_state.vectorstore = create_vectorstore(docs)
    st.success("PDFs processed and vector store created!")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Get LLM and its type
        llm, llm_type = get_llm(st.session_state.gemini_api_key, st.session_state.hf_api_key)
        
        # Retrieve relevant docs if vectorstore exists
        relevant_docs = []
        if st.session_state.vectorstore:
            relevant_docs = retrieve_relevant_docs(st.session_state.vectorstore, user_input)
        
        # Decide RAG or pure LLM
        if relevant_docs:
            response = generate_rag_response(llm, llm_type, user_input, relevant_docs, st.session_state.chat_history)
        else:
            response = generate_pure_llm_response(llm, llm_type, user_input, st.session_state.chat_history)
        
        st.markdown(response)
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})