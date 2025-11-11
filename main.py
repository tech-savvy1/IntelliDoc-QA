import os
import shutil
from uuid import uuid4
from dotenv import load_dotenv
import uvicorn

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

# ---------------- ENV & CONFIG ----------------
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY must be set in your .env file.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIRECTORY, exist_ok=True)

# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="IntelliDoc-QA: Advanced RAG System 🤖📄",
    description="Upload a PDF, index it to ChromaDB, and ask questions with cited sources.",
    version="2.0.0",
)

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/playground")

# ---------- Simple UI ----------
@app.get("/playground", response_class=HTMLResponse, include_in_schema=False)
def playground():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>IntelliDoc-QA Playground</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen">
  <div class="max-w-3xl mx-auto p-6">
    <header class="mb-6">
      <h1 class="text-3xl font-semibold tracking-tight">IntelliDoc-QA</h1>
      <p class="text-slate-400">Upload a PDF → ask a question → get a concise answer with sources.</p>
    </header>

    <div class="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl">
      <div>
        <label class="block text-sm font-medium mb-2">1) Upload PDF</label>
        <div class="flex items-center gap-3">
          <input id="pdfFile" type="file" accept="application/pdf"
                 class="file:mr-4 file:py-2 file:px-4 file:rounded-xl file:border-0
                        file:text-sm file:font-medium file:bg-indigo-500 file:text-white
                        hover:file:bg-indigo-600 text-slate-300 w-full"/>
          <button id="uploadBtn"
                  class="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50">
            Upload & Index
          </button>
        </div>
        <p id="uploadStatus" class="text-sm mt-2 text-slate-400"></p>
      </div>

      <hr class="my-6 border-slate-800">

      <div>
        <label class="block text-sm font-medium mb-2">2) Ask a question</label>
        <div class="flex gap-3">
          <input id="question" type="text" placeholder="e.g., Summarize section 2…"
                 class="flex-1 rounded-xl bg-slate-800 border border-slate-700 px-4 py-2
                        placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-600"/>
          <button id="askBtn"
                  class="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50" disabled>
            Ask
          </button>
        </div>
        <p id="askHint" class="text-xs mt-2 text-slate-400">Upload a PDF first to enable asking.</p>
      </div>

      <div id="result" class="mt-6 hidden">
        <h3 class="font-semibold mb-2">Answer</h3>
        <div id="answer"
             class="rounded-xl border border-slate-800 bg-slate-900/70 p-4 leading-relaxed"></div>

        <details class="mt-4">
          <summary class="cursor-pointer text-slate-300">Sources</summary>
          <div id="sources" class="mt-2 space-y-3"></div>
        </details>
      </div>
    </div>

    <footer class="text-xs text-slate-500 mt-6">Built with FastAPI • LangChain • ChromaDB</footer>
  </div>

  <script>
    async function parseResponse(res) {
      const ct = res.headers.get('content-type') || '';
      if (ct.includes('application/json')) return { data: await res.json(), isJson: true };
      return { data: await res.text(), isJson: false };
    }

    const uploadBtn = document.getElementById('uploadBtn');
    const pdfFile = document.getElementById('pdfFile');
    const uploadStatus = document.getElementById('uploadStatus');
    const askBtn = document.getElementById('askBtn');
    const askHint = document.getElementById('askHint');
    const questionInput = document.getElementById('question');
    const resultBlock = document.getElementById('result');
    const answerDiv = document.getElementById('answer');
    const sourcesDiv = document.getElementById('sources');

    let collectionName = null;

    uploadBtn.addEventListener('click', async () => {
      const file = pdfFile.files[0];
      if (!file) {
        uploadStatus.textContent = "Please choose a PDF file.";
        uploadStatus.className = "text-sm mt-2 text-rose-400";
        return;
      }
      uploadBtn.disabled = true;
      uploadStatus.textContent = "Uploading & indexing…";
      uploadStatus.className = "text-sm mt-2 text-slate-400";

      const form = new FormData();
      form.append('file', file);

      try {
        const res = await fetch('/upload-and-index-pdf/', { method: 'POST', body: form });
        const { data, isJson } = await parseResponse(res);
        if (!res.ok) throw new Error(isJson && data?.detail ? data.detail : (data || 'Upload failed'));
        collectionName = data.collection_name;
        uploadStatus.textContent = `Ready ✅  Collection: ${collectionName}`;
        uploadStatus.className = "text-sm mt-2 text-emerald-400";
        askBtn.disabled = false;
        askHint.textContent = "Ask anything about your PDF.";
      } catch (e) {
        uploadStatus.textContent = e.message;
        uploadStatus.className = "text-sm mt-2 text-rose-400";
      } finally {
        uploadBtn.disabled = false;
      }
    });

    askBtn.addEventListener('click', async () => {
      const q = questionInput.value.trim();
      if (!q || !collectionName) return;
      askBtn.disabled = true;
      askBtn.textContent = "Thinking…";
      resultBlock.classList.remove('hidden');
      answerDiv.textContent = "Working on it…";
      sourcesDiv.innerHTML = "";

      const form = new URLSearchParams();
      form.append('collection_name', collectionName);
      form.append('query', q);

      try {
        const res = await fetch('/ask-question-with-sources/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: form
        });
        const { data, isJson } = await parseResponse(res);
        if (!res.ok) throw new Error(isJson && data?.detail ? data.detail : (data || 'Failed to get answer'));
        answerDiv.textContent = data.answer || "(no answer)";
        sourcesDiv.innerHTML = (Array.isArray(data.sources) ? data.sources : []).map((s, i) => `
          <div class="rounded-lg bg-slate-800/60 border border-slate-700 p-3">
            <div class="text-xs text-slate-400 mb-1">Source ${i+1} • Page ${s.page_number ?? 'N/A'}</div>
            <div class="text-sm whitespace-pre-wrap">${(s.content || '').replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}</div>
          </div>
        `).join('');
      } catch (e) {
        answerDiv.textContent = e.message;
      } finally {
        askBtn.disabled = false;
        askBtn.textContent = "Ask";
      }
    });
  </script>
</body>
</html>
    """

# ---------------- RAG ENDPOINTS ----------------

@app.post("/upload-and-index-pdf/")
async def upload_and_index_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    # Stable collection name from original filename
    collection_name = os.path.splitext(file.filename)[0].replace(" ", "_").lower()
    persist_directory = os.path.join(CHROMA_DB_DIRECTORY, collection_name)

    # Fresh index for this filename
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory, ignore_errors=True)

    # ---- use a UNIQUE temp path to avoid WinError 32 ----
    temp_file_path = os.path.join(UPLOADS_DIR, f"{uuid4().hex}.pdf")

    try:
        # Write upload to a unique file (no collision with example.pdf opened elsewhere)
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Load, split, embed, persist
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        return {
            "status": "success",
            "filename": file.filename,
            "collection_name": collection_name,
            "message": "PDF processed and ready for Q&A.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    finally:
        # Ensure we always release/delete the temp file (prevents Windows file locks)
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception:
            pass

@app.post("/ask-question-with-sources/")
async def ask_question_with_sources(
    collection_name: str = Form(...),
    query: str = Form(...)
):
    persist_directory = os.path.join(CHROMA_DB_DIRECTORY, collection_name)
    if not os.path.exists(persist_directory):
        raise HTTPException(status_code=404, detail="Document collection not found. Upload the PDF first.")

    try:
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        system_prompt = """
        You are an assistant for question-answering tasks.
        Use the retrieved context to answer. If you don't know, say you don't know.
        Keep answers concise (max 3 sentences).

        Context:
        {context}
        """
        prompt = PromptTemplate.from_template(system_prompt)

        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, qa_chain)

        result = await rag_chain.ainvoke({"input": query})

        sources = [
            {"content": doc.page_content, "page_number": doc.metadata.get("page", "N/A")}
            for doc in result["context"]
        ]

        return {"query": query, "answer": result["answer"], "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

# ---------------- ENTRYPOINT ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
