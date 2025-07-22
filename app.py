from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from typing import List
import google.generativeai as genai
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from google.api_core import retry
import os
from fastapi.responses import JSONResponse

os.environ["GOOGLE_API_KEY"] =  os.getenv("GOOGLE_API_KEY")  
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def _init_(self, document_mode=True):
        self.document_mode = document_mode

    def _call_(self, input: Documents) -> Embeddings:
        task_type = "retrieval_document" if self.document_mode else "retrieval_query"
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}

        response = genai.embed_content(
            model="models/text-embedding-004",
            content=input,
            task_type=task_type,
            request_options=retry_policy,
        )
        return response["embedding"]

DB_NAME = "pdf_qa_db"
embed_fn = GeminiEmbeddingFunction(document_mode=True)
chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

@app.get("/")
async def root():
    return JSONResponse({"message": "Hello from FastAPI on Vercel! by Yash Patil"})

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    full_text = ""
    for file in files:
        reader = PdfReader(file.file)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                full_text += content + "\n"

    if not full_text.strip():
        return {"message": "No text could be extracted from the uploaded PDFs."}

    embed_fn.document_mode = True
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    db.add(documents=[full_text], ids=[str(len(db.get()["ids"]))])

    return {"message": "PDFs uploaded and content embedded successfully."}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    embed_fn.document_mode = False
    result = db.query(query_texts=[question], n_results=1)

    if not result["documents"] or not result["documents"][0]:
        return {"answer": "No relevant passage found in the uploaded PDFs."}

    passage = result["documents"][0][0]
    passage_oneline = passage.replace("\n", " ")
    query_oneline = question.replace("\n", " ")

    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
    strike a friendly and conversational tone. If the passage is irrelevant to the answer, you may ignore it.

    QUESTION: {query_oneline}
    PASSAGE: {passage_oneline}
    """

    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return {"answer": response.text}