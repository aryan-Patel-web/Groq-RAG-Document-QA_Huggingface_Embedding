
# Groq-RAG-Document-QA

📄 **Groq-RAG-Document-QA** is a powerful Retrieval-Augmented Generation (RAG) app built with Streamlit.  
It combines Groq’s blazing-fast LLaMA3 LLM with HuggingFace embeddings and FAISS vector search to answer questions from PDF research papers or documents.

---

## 🚀 Features

✅ Upload a directory of PDF files for vector indexing  
✅ Ask natural language questions about your documents  
✅ Uses Groq’s LLaMA3 model for context-based answers  
✅ Displays matching document chunks for transparency  
✅ Fast response times and easy local deployment

---

## 🌟 Benefits

- Perfect for research, legal, financial, or academic document exploration
- No need to manually read large PDFs
- Supports local or cloud deployment
- Extensible for other file formats or larger corpora

---

## 🗂️ Project Structure

├── research_papers/ # Your PDFs for indexing
├── .env
├── streamlit_app.py
├── requirements.txt
└── README.md



---

## ⚙️ Requirements

- Python 3.9+
- Streamlit
- langchain
- langchain_groq
- langchain_huggingface
- faiss-cpu
- python-dotenv

Install all dependencies:

```bash
pip install -r requirements.txt


🔑 Environment Variables
Create a .env file:

GROK_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here

▶️ Run the App

streamlit run streamlit_app.py


💬 Example Prompt

"What are the main findings about neural networks in the uploaded research papers?"


👨‍💻 Author
Aryan Patel