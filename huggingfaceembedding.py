import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables
load_dotenv()
os.environ['GROK_API_KEY'] = os.getenv("GROK_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Page title
st.title("📄 RAG Document Q&A with Groq + LLaMA3")

# Create embeddings safely with device specified to avoid 'meta tensor' error
def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Change to "cuda" if using GPU
    )

# LLM using Groq (LLaMA3)
llm = ChatGroq(groq_api_key=os.environ['GROK_API_KEY'], model_name="Llama3-8b-8192")

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

<context>
{context}
</context>

Question: {input}
""")

# Vector database creator
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = create_embeddings()
        loader = PyPDFDirectoryLoader("research_papers")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = splitter.split_documents(docs[:50])  # Load only first 50 for speed

        st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
        st.session_state.docs_loaded = True

# UI input
user_prompt = st.text_input("🔍 Enter your query based on the research papers:")

# Embedding button
if st.button("📂 Embed Documents"):
    create_vector_embedding()
    st.success("✅ Vector database created successfully!")

# Query handling
if user_prompt and "vectors" in st.session_state:
    retriever = st.session_state.vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    duration = time.process_time() - start

    st.subheader("📬 Answer")
    st.write(response['answer'])
    st.caption(f"⏱️ Response Time: {duration:.2f} seconds")

    with st.expander("🔎 Document Similarity Context"):
        for i, doc in enumerate(response.get('context', [])):
            st.markdown(f"**Match {i+1}:**")
            st.write(doc.page_content)
            st.write("---")

elif user_prompt:
    st.warning("⚠️ Please embed the documents first by clicking '📂 Embed Documents'.")

