# 💡 IntelliDoc-QA  

**IntelliDoc-QA** is an AI-powered document question-answering system built with **FastAPI** and **LangChain**.  
It lets you upload a PDF, automatically index it locally with **ChromaDB**, and ask natural-language questions through a simple, built-in web interface.

---

## ✨ Features  
- 📄 Upload and process PDF files  
- 🔍 Split text into chunks and embed them locally  
- 🧠 Query using OpenAI’s GPT models via LangChain  
- 💾 Uses **ChromaDB** as the local vector database (no external services)  
- 💻 Beautiful built-in web UI (Tailwind + FastAPI)  
- ⚙️ Works completely locally — no Pinecone or Jinja2 setup required  

---

## 🛠️ Tech Stack  
- **Backend:** FastAPI + Uvicorn  
- **LLM:** OpenAI (via `langchain-openai`)  
- **Vector Database:** ChromaDB  
- **Embeddings:** Sentence-Transformers (MiniLM-L6-v2)  
- **PDF Processing:** LangChain Community’s `PyPDFLoader`  
- **Frontend:** HTML + TailwindCSS (served directly by FastAPI)  
- **Environment:** Python 3.10+  

---

## 📸 Preview  

After running the app, open [http://127.0.0.1:8000/playground](http://127.0.0.1:8000/playground):

![Web App](IntelliDocQA1.png)
![Web App](IntelliDocQA2.png)
![Web App](IntelliDocQA3.png)
![Web App](IntelliDocQA4.png)

---

## 🚀 Getting Started  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/tech-savvy1/IntelliDoc-QA.git
cd IntelliDoc-QA
```

### 2️⃣ Set up a virtual environment  
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt --upgrade
```

### 4️⃣ Create a `.env` file in the root directory  
```
OPENAI_API_KEY=your_openai_api_key
```

💡 *No Pinecone keys are needed — ChromaDB stores everything locally.*

---

## ▶️ Run the App  
Start the FastAPI server:
```bash
uvicorn main:app --reload
```

Then open in your browser:
👉 [http://127.0.0.1:8000/playground](http://127.0.0.1:8000/playground)

---

## 💬 Usage Flow  
1. **Upload a PDF** — the app extracts text and indexes it in ChromaDB.  
2. **Ask questions** — enter a natural-language question related to the document.  
3. **Get concise answers** — GPT analyzes the relevant chunks and responds.  
4. **View sources** — click “Sources” to expand and see which pages were used.  

---

## 📂 Project Structure  
```
IntelliDoc-QA/
│
├── main.py              # FastAPI app + web UI + API endpoints
├── requirements.txt     # Dependencies
├── .env                 # Environment variables (API key)
├── chroma_db/           # Local vector database (auto-created)
├── uploads/             # Temporary uploaded files
└── README.md            # This file
```

---

## 🧰 Key Endpoints  

| Method | Endpoint | Description |
|:-------|:----------|:-------------|
| `GET` | `/playground` | Web UI to upload & query PDFs |
| `POST` | `/upload-and-index-pdf/` | Uploads and indexes a PDF |
| `POST` | `/ask-question-with-sources/` | Asks a question about the indexed document |

---

## 🧹 Troubleshooting  

**🔒 “WinError 32”**  
If you see:  
```
PermissionError: [WinError 32] The process cannot access the file because it is being used by another process
```
→ Fixed in the latest version — uploads are written to unique temp files and auto-deleted.

**🧠 No answers or empty sources?**  
- Ensure your `.env` file contains a valid `OPENAI_API_KEY`.  
- Try with a text-heavy PDF.  
- Check the console for LangChain logs.

---

## 🧑‍💻 Credits  
Built by **Lashiya Kashyap** using:
- FastAPI ⚡
- LangChain 🧠
- ChromaDB 💾
- Tailwind 💅
