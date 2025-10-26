import streamlit as st
import os
from io import BytesIO
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- Configuration ---
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001" # Standard Gemini embedding model

st.set_page_config(
    page_title="PDF RAG App (Gemini)",
    page_icon="📚",
    layout="wide"
)

# --- Core Utility Functions ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            # Use BytesIO to handle the uploaded file object
            pdf_reader = PdfReader(BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {e}")
            return None
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vector_store(text_chunks, api_key):
    """
    Creates an in-memory FAISS vector store from text chunks.
    This function is cached to prevent re-running on every interaction.
    """
    if not api_key:
        st.error("Cannot create vector store without API Key.")
        return None
    try:
        # Use the Gemini embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store (likely an API key issue): {e}")
        return None

def get_conversational_chain(api_key):
    """Defines the RAG chain for question answering."""
    prompt_template = """
    Answer the user's question only based on the provided context.
    If the answer is not found in the context, politely state that the information
    is not available in the provided documents.

    Context:
    {context}

    Question:
    {question}
    """
    
    # Use ChatGoogleGenerativeAI for the LLM
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3, google_api_key=api_key)

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Load the QA chain
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    
    return chain

def user_input(user_question, vector_store, api_key):
    """Handles the user's question, performs retrieval, and calls the RAG chain."""
    if not vector_store:
        st.error("Vector store is not initialized. Please process the PDFs first.")
        return

    try:
        # 1. Retrieval: Find the most relevant chunks (top 4)
        docs = vector_store.similarity_search(user_question)

        # 2. Generation: Pass the retrieved documents and question to the chain
        chain = get_conversational_chain(api_key)

        with st.spinner("Generating response..."):
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
        
        st.session_state.chat_history.append(("Question", user_question))
        st.session_state.chat_history.append(("Answer", response["output_text"]))

    except Exception as e:
        st.error(f"An error occurred during generation: {e}")
        st.info("Ensure your API key is correct and the documents were uploaded successfully.")

# --- Streamlit UI ---

st.header("📚 Gemini PDF-Based RAG Application")

# --- API Key Handling (Sidebar) ---

st.sidebar.title("Configuration")
# Try to load API key from Streamlit secrets (for deployment)
api_key_from_secrets = st.secrets.get("GEMINI_API_KEY")

if api_key_from_secrets:
    api_key = api_key_from_secrets
    st.sidebar.success("API Key loaded from Streamlit Secrets.")
    # Use a hidden placeholder if key is from secrets
    api_key_input = st.sidebar.empty() 
else:
    # Fallback to sidebar input (for local testing)
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Paste your Google AI Studio API Key here. For deployment, use Streamlit Secrets (`GEMINI_API_KEY`)."
    )

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Document Processing ---

with st.sidebar.expander("1. Upload & Process PDFs", expanded=True):
    pdf_docs = st.file_uploader(
        "Upload your PDF Documents", 
        accept_multiple_files=True,
        type=["pdf"]
    )
    
    if st.button("Process Documents", use_container_width=True):
        if pdf_docs and api_key:
            with st.spinner("Processing... This may take a moment."):
                # Clear existing cache/store if new documents are processed
                st.cache_resource.clear()
                st.session_state.vector_store = None 
                st.session_state.chat_history = []

                # 1. Get PDF text
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.error("Could not extract text from PDFs.")
                else:
                    # 2. Get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # 3. Create vector store
                    st.session_state.vector_store = get_vector_store(text_chunks, api_key)
                    
                    if st.session_state.vector_store:
                        st.success(f"Processing complete! {len(text_chunks)} chunks created and ready for QA.")
                    else:
                        st.error("Failed to create vector store. Check API key and logs.")

        elif not api_key:
            st.warning("Please enter your Gemini API Key first.")
        else:
            st.warning("Please upload PDF documents to process.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**LLM:** `{LLM_MODEL}`\n\n**Embedding:** `{EMBEDDING_MODEL}`")


# --- Main Chat Interface ---

st.subheader("2. Chat with Your Documents")

user_question = st.text_input(
    "Ask a question based on your uploaded documents:",
    placeholder="e.g., What are the key findings discussed on page 5?",
    key="user_question_input"
)

if user_question and st.session_state.vector_store:
    user_input(user_question, st.session_state.vector_store, api_key)
elif user_question and not st.session_state.vector_store:
    st.warning("Please upload and process your PDF documents first in the sidebar.")

# --- Display Chat History ---
st.subheader("Conversation History")
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.info("Start by uploading PDFs and asking a question!")
    else:
        for role, text in st.session_state.chat_history:
            if role == "Question":
                st.markdown(f"**User**: *{text}*")
            else:
                st.markdown(f"**🤖 Gemini**: {text}")
