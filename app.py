import os, tempfile, shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

# Load API key
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Set up Streamlit page
st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("🧠 Ask Questions About Your PDF")
st.markdown("Upload a PDF and ask any question. Hybrid search + LLM (LLaMA3-70B)")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# Global states
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "tmp_dir" not in st.session_state:
    st.session_state.tmp_dir = None

# Define lowercased BM25 retriever
class LowercaseBM25Retriever(BM25Retriever):
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        **kwargs,
    ):
        return super()._get_relevant_documents(
            query.lower(),
            run_manager=run_manager,
            **kwargs
        )

# Process PDF on button click
if uploaded_file and st.button("📄 Process PDF"):
    with st.spinner("Loading PDF..."):
        tmp_dir = tempfile.mkdtemp()
        file_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        doc_loader = PyPDFLoader(file_path)
        doc_pdf = doc_loader.load()
        print(type(doc_pdf))

    with st.spinner("Splitting text..."):
        text_splitter_recursive = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
        docs_recursive = text_splitter_recursive.split_documents(doc_pdf)
        texts = [doc.page_content.lower() for doc in docs_recursive]

    with st.spinner("Creating embeddings and vectorstore..."):
        embedding = HuggingFaceBgeEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = Chroma.from_documents(
            documents=docs_recursive,
            embedding=embedding,
            persist_directory=None  # In-memory DB
        )
        chroma_retr = vectordb.as_retriever(search_kwargs={"k": 2})

    with st.spinner("Creating BM25 & Hybrid retriever..."):
        bm25_retr = LowercaseBM25Retriever.from_texts(texts)
        bm25_retr.k = 2
        ensemble_retr = EnsembleRetriever(retrievers=[bm25_retr, chroma_retr], weights=[0.3, 0.7])

    with st.spinner("Initializing LLM & chain..."):
        prompt = ChatPromptTemplate.from_template(
            """
Answer the following question only with information found in the context.
Search the entire context before answering.
Give a long, detailed answer.

After answering, suggest two follow-up queries the user might find useful,
formatted exactly as a numbered list.

<context>
{context}
</context>

# Question: {input}
"""
        )

        llm = ChatGroq(
            temperature=0.5,
            model_name="llama3-70b-8192",
            groq_api_key=groq_api_key
        )

        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(ensemble_retr, doc_chain)

        st.session_state.rag_chain = rag_chain
        st.session_state.tmp_dir = tmp_dir

    st.success("✅ PDF processed successfully! You can now ask questions.")

# Ask block with input + button
if st.session_state.rag_chain:
    st.markdown("---")
    st.subheader("Ask your question")
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("Write your question here", key="query_input")
    with col2:
        ask_button = st.button("Ask")

    if ask_button and query:
        with st.spinner("🤖 Generating answer..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": query})
                st.markdown("### 📘 Answer")
                st.markdown(response['answer'])
            except Exception as e:
                st.error(f"Error: {e}")

# Cleanup temporary files
def cleanup():
    if st.session_state.get("tmp_dir") and os.path.exists(st.session_state.tmp_dir):
        shutil.rmtree(st.session_state.tmp_dir)

import atexit
atexit.register(cleanup)
