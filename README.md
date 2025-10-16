# 🦙 Llama4-RAG-DocQA — Document Question Answering App
**Llama 4** Docx Chatter is an interactive **Retrieval-Augmented Generation (RAG)** app built with **LangChain**, **Groq Llama 4**, and **Gradio**.
It allows users to upload **.docx** files (or a .zip containing multiple documents), automatically builds a temporary **vector** database, and enables conversational Q&A over the document contents using **Llama 4-Scout 17B 16E Instruct**.

---

## 🧠 Overview
  * Upload **.docx** or **.zip** files.
  * The app extracts, chunks, and embeds the text using **Hugging Face Embeddings**.
  * Embeddings are stored in an **In-Memory Vector Store** for fast retrieval.
  * User questions are answered using a **Retrieval-Augmented Generation chain powered by Llama 4 (Groq API)**.
  * Everything runs locally in a **Gradio** chat interface.

---
## 📸 Preview

<img width="959" height="880" alt="Screenshot from 2025-10-16 10-14-52" src="https://github.com/user-attachments/assets/dfa4ee3c-8e77-4a51-a29c-18fc2303a87f" />

---

## ✨ Features

  - ✅ Upload single or multiple documents (.docx / .zip)
  - ✅ Automatic text splitting and vector embedding
  - ✅ In-Memory vector DB for quick retrieval
  - ✅ Detailed and grounded answers based on context
  - ✅ Reset session and upload new files instantly
  - ✅ Interactive UI built with Gradio (SOFT theme)
  
---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Language** | Python 3.9+ |
| **LLM Backend** | Groq API (Llama-4-Scout 17B-16E Instruct) |
| **Embeddings** | Hugging Face – mixedbread-ai/mxbai-embed-large-v1 |
| **Framework** | [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://www.langchain.com/) | **Frontend** | Gradio UI for chat interaction |
| **Vector Store** | InMemoryVectorStore (LangChain Core) |

---

**Install Dependencies:**  
```bash
pip install -r requirements.txt
```

**Create a .env file:**  
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

**Run the app:**  
```bash
python main.py
```
---

## 🧩 File Structure

```bash
.
Llama4-RAG-DocQA/
│
├── Lama4_RAG.py            # main Gradio app file
├── main.py                 # optional helper entry
├── Vectordb/               # temporary vector store
├── docx/                   # uploaded documents
├── .env                    # API keys (not pushed to GitHub)
├── .gitignore              # ignore .env, Vectordb, docx etc.
└── README.md
```



