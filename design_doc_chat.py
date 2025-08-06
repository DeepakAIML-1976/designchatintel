import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# ----------------------------
# Constants
# ----------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4"
PERSIST_DIRECTORY = "db"
CHROMA_COLLECTION_NAME = "design_doc_collection"

# ----------------------------
# Initialize models
# ----------------------------
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="📄 Design Document Intelligence Chat")
st.title("📄 Design Document Intelligence Chat")

uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, XLSX)", 
    type=["pdf", "docx", "xlsx"], 
    accept_multiple_files=True
)

# Session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chain" not in st.session_state and uploaded_files:
    all_docs = []

    def load_document(file_path, file_type):
        if file_type == "pdf":
            return PyMuPDFLoader(file_path).load()
        elif file_type == "docx":
            return UnstructuredWordDocumentLoader(file_path).load()
        elif file_type == "xlsx":
            return UnstructuredExcelLoader(file_path).load()
        else:
            return []

    for file in uploaded_files:
        suffix = file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        docs = load_document(temp_file_path, suffix)
        all_docs.extend(docs)

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_docs)

    # Vector DB
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    retriever = vectordb.as_retriever()

    # Add memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    st.session_state.chain = chain

# User chat interface
if "chain" in st.session_state:
    user_query = st.chat_input("Ask a question about your documents:")

    if user_query:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.run(user_query)
            st.session_state.chat_history.append(("user", user_query))
            st.session_state.chat_history.append(("ai", response))

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)
