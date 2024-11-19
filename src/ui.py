from app import Chatbot, stream_output_to_streamlit
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import streamlit as st
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
st.session_state["llm"] = ChatOllama(model="llama3.2:1b", base_url=OLLAMA_URL)

st.title("Chatbot with LLM and RAG 🤖")

with st.sidebar:
    st.header("Upload PDFs")
    
    # File uploader in the sidebar
    uploaded_files = st.file_uploader(
        "Upload up to 3 PDF files", 
        type="pdf", 
        accept_multiple_files=True, 
        key="pdf_uploader"
    )
    
    # Button to load PDFs
    load_button = st.button("Index Documents", key="load_pdfs")

# Process PDFs and store them in Chroma when the button is clicked
if load_button:
    if uploaded_files:
        if len(uploaded_files) <= 3:
            st.sidebar.success(f"{len(uploaded_files)} PDF(s) uploaded successfully!")
            
            documents = []  # To store the parsed documents
            temp_file_paths = []  # To keep track of the temporary file paths for deletion
            
            for uploaded_file in uploaded_files:
                # Temporarily save the uploaded file
                temp_file_path = os.path.join("./", uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                temp_file_paths.append(temp_file_path)  # Add the file path to the list

                # Load the PDF content using PyPDFLoader
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())  # Extend the list with loaded documents

            with st.spinner(
                    "Creating embeddings and loading documents into Chroma..."
                ):
            # Store documents in Chroma database
                st.session_state["db"] = Chroma.from_documents(
                    documents,
                    OllamaEmbeddings(model="nomic-embed-text"),
                )

                st.session_state["app"] = Chatbot(db=st.session_state["db"], llm=st.session_state["llm"]).create_app()
                st.session_state["config"] = {"configurable": {"thread_id": "abc123"}}
            
            st.info("All set to answer questions!")
            
            # Delete the temporary files after loading
            for temp_file_path in temp_file_paths:
                os.remove(temp_file_path)  # Remove the temporary file
            
        else:
            st.sidebar.error("You have uploaded more than 3 files. Please upload up to 3 files only.")
    else:
        st.sidebar.warning("No files uploaded yet.")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = stream_output_to_streamlit(prompt, st.session_state["app"], st.session_state["config"])
        st.session_state.messages.append({"role": "assistant", "content": response})
