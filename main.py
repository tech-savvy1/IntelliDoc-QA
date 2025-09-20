from dotenv import load_dotenv
load_dotenv()

import os
import hashlib
import logging
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic_settings import BaseSettings

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION & SETUP ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables and validate using Pydantic
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str = "intellidoc-qa"

    class Config:
        env_file = ".env"

try:
    settings = Settings()
except Exception as e:
    logging.error(f"Failed to load settings: {e}")
    raise

# Initialize LLM and Embeddings
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, streaming=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 2. FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="IntelliDoc QA",
    description="An interactive API to chat with your PDF documents.",
    version="2.0.0"
)
templates = Jinja2Templates(directory="templates")


# --- 3. CORE RAG LOGIC & API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page for the chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF, chunks it, creates embeddings, and stores them in Pinecone
    using a unique namespace derived from the file's content.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    try:
        # Read file content to generate a unique ID (hash) and for processing
        file_content = await file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        temp_file_path = f"./temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        # 1. Load the document
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # 2. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # 3. Create and store embeddings in Pinecone with a specific namespace
        logging.info(f"Creating embeddings for {len(docs)} chunks in namespace: {file_hash}")
        PineconeVectorStore.from_documents(
            docs, embeddings, index_name=settings.PINECONE_INDEX_NAME, namespace=file_hash
        )

        os.remove(temp_file_path)
        
        logging.info(f"PDF '{file.filename}' processed successfully. Document ID: {file_hash}")
        return {"status": "success", "doc_id": file_hash, "filename": file.filename}

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")


@app.post("/ask")
async def ask_question(doc_id: str = Form(...), query: str = Form(...)):
    """
    Asks a question about a previously uploaded PDF using its unique doc_id.
    Streams the answer back to the client.
    """
    if not doc_id or not query:
        raise HTTPException(status_code=400, detail="doc_id and query are required.")
    
    try:
        logging.info(f"Received query for doc_id '{doc_id}': '{query}'")
        
        # 1. Get the existing vector store from Pinecone using the namespace
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=doc_id
        )
        retriever = vector_store.as_retriever()

        # 2. Define the prompt template
        prompt_template = """
        You are a helpful assistant. Answer the user's question based ONLY on the following context.
        If the answer is not found in the context, say: "I couldn't find an answer to that in the document."
        
        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # 3. Create the RAG chain for streaming
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 4. Stream the response
        async def stream_response():
            async for chunk in rag_chain.astream(query):
                yield chunk

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    except Exception as e:
        logging.error(f"Error during Q&A for doc_id {doc_id}: {e}")
        # Cannot return StreamingResponse with error, so we raise HTTPException
        # The client-side will need to handle the non-200 response.
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# --- 4. RUN THE APP ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)