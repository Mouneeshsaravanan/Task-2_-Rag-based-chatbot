
#  RAG based Chat Bot(Techjays Task 2)

A Retrieval-Augmented Generation (RAG) chat app built with Streamlit. Upload one or more PDF files; the app indexes them with FAISS and answers questions using Google Gemini or, if unavailable, a Hugging Face Inference API fallback.

## Features
- Upload multiple PDFs; automatic chunking with overlap
- FAISS vector store using `all-MiniLM-L6-v2` sentence embeddings
- Hybrid LLM backend:
  - Primary: Google Gemini (`gemini-1.5-flash`)
  - Fallback: Hugging Face Inference API (`mistralai/Mixtral-8x7B-Instruct-v0.1`)
- Chat history maintained in Streamlit session state
- Uses RAG when relevant chunks exist; falls back to pure LLM when not
- One-click reset for chat history and vector store

## Project Structure

```
app.py                # Streamlit UI and main app flow
document_loader.py    # PDF loading and text splitting
vector_store.py       # Embeddings + FAISS vector store and retrieval
llm_handler.py        # LLM initialization and response generation (RAG and pure LLM)
requirements.txt      # Python dependencies
```

## Prerequisites
- Python 3.10+
- OS: Windows/macOS/Linux
- API keys (any one or both):
  - Google Gemini API key (recommended)
  - Hugging Face Inference API token (fallback)

## Quick Start

1) Create and activate a virtual environment

Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux (bash):
```
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

3) Set environment variables in a `.env` file at the project root
```
GEMINI_API_KEY=your_gemini_key_here
HF_API_KEY=your_huggingface_token_here
```
- You can supply only `GEMINI_API_KEY`. If Gemini fails (e.g., rate limit), the app uses `HF_API_KEY` if present.

4) Run the app
```
streamlit run app.py
```

5) Use the UI
- Click “Upload PDFs” to select one or more `.pdf` files
- Ask a question in the chat input
- Click “Clear Chat History and Data” to reset

## How It Works
- `document_loader.load_and_split_documents`
  - Saves uploads to temporary files
  - Loads with `PyPDFLoader`
  - Splits into 1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`
- `vector_store.create_vectorstore`
  - Builds a FAISS index using `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`)
- Retrieval
  - `retrieve_relevant_docs` fetches up to k=5 similar chunks and filters by a similarity `threshold` (default 0.5)
- LLM selection and generation
  - Tries Gemini first via `langchain-google-genai` (`gemini-1.5-flash`)
  - On invalid credentials, rate limits, or errors, falls back to Hugging Face Inference API (`mistralai/Mixtral-8x7B-Instruct-v0.1`)
  - RAG path builds a prompt with prior chat history (excluding current turn) and retrieved context
  - Pure LLM path uses only chat history and the new user query

## Configuration Notes
- Environment loading via `python-dotenv` (`load_dotenv()` in `app.py`)
- Retrieval threshold can be tuned in `vector_store.retrieve_relevant_docs`
- Generation parameters
  - HF: `max_new_tokens=500`, `temperature=0.7`
  - Gemini uses a simple `LLMChain` with the constructed prompt

## Troubleshooting
- Startup/auth issues
  - Check `.env` values and internet connectivity
  - `pip install --upgrade pip` then reinstall requirements
- Gemini quota/invalid key
  - Console logs indicate fallback to Hugging Face if `HF_API_KEY` is set
- PDF text not extracted
  - `PyPDFLoader` works on digital PDFs. Scanned PDFs may need OCR (not included)
- Irrelevant answers
  - Lower/raise `threshold` in `retrieve_relevant_docs`
  - Try a domain-specific embedding model

## Security & Privacy
- Uploaded PDFs are written to temp files during processing and deleted afterward
- No persistent DB; chat and vector store live in Streamlit session memory
- Keep API keys secret; do not commit `.env`

## Extending
- Change embeddings: edit `model_name` in `vector_store.py`
- Adjust chunking: edit `chunk_size`/`chunk_overlap` in `document_loader.py`
- Swap LLMs or prompts: edit `llm_handler.py`
- Persist vectors: add FAISS index save/load between sessions

## Requirements
Key libraries in `requirements.txt`:
- streamlit
- google-generativeai, langchain-google-genai
- langchain, langchain-community
- langchain-huggingface, huggingface_hub
- sentence-transformers, faiss-cpu
- pypdf, python-dotenv

## License
Add your preferred license (e.g., MIT).


OUTPUT

<img width="980" height="877" alt="image" src="https://github.com/user-attachments/assets/84bb6c27-cedd-4600-932b-f83b6bed14d3" />
<img width="1027" height="906" alt="image" src="https://github.com/user-attachments/assets/4ae34e99-ae33-4d06-9aec-5c9376936687" />
<img width="1082" height="896" alt="image" src="https://github.com/user-attachments/assets/ca345899-b19b-459f-a585-12777e33e029" />
