from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

app = FastAPI()

# Load and process documents once on startup
try:
    loader = TextLoader("data.txt")
    documents = loader.load()
    print("📄 Loaded Documents:", documents)

    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_documents(documents)
    print("🧩 Split Chunks:", [chunk.page_content for chunk in chunks])

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    llm = Ollama(model="llama3")  # or "llama3:instruct" if needed

    vectorstore = FAISS.from_documents(chunks, embedding)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("✅ Vector store created and QA chain ready")

except Exception as e:
    print(f"❌ Error during startup: {e}")
    qa_chain = None

# Pydantic request model
class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
    <head>
        <title>RAG Chat UI</title>
        <style>
            body { font-family: Arial; max-width: 600px; margin: auto; padding-top: 50px; }
            .chat-box { margin-top: 20px; }
            .user-msg { font-weight: bold; }
            .response { background: #f1f1f1; padding: 10px; border-radius: 5px; margin-top: 5px; }
        </style>
    </head>
    <body>
        <h2>Chat with your Document (RAG + Ollama)</h2>
        <input type="text" id="query" placeholder="Ask a question..." onkeydown="if(event.key==='Enter')sendQuery()" style="width: 80%;">
        <button onclick="sendQuery()">Send</button>
        <div class="chat-box" id="chat"></div>

        <script>
            async function sendQuery() {
                const question = document.getElementById("query").value;
                document.getElementById("chat").innerHTML += `<div class="user-msg">You: ${question}</div>`;
                const response = await fetch("/ask", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: question })
                });
                const result = await response.json();
                document.getElementById("chat").innerHTML += `<div class="response">AI: ${result.response}</div>`;
                document.getElementById("query").value = "";
            }
        </script>
    </body>
    </html>
    """

@app.post("/ask")
async def ask_doc(req: QueryRequest):
    if qa_chain is None:
        return JSONResponse(content={"response": "Server is not ready. Check logs for errors."})
    try:
        response = qa_chain.run(req.query)
        return JSONResponse(content={"response": response})
    except Exception as e:
        return JSONResponse(content={"response": f"Error during processing: {str(e)}"})

@app.get("/debug")
def debug():
    return {
        "documents": [doc.page_content for doc in documents] if 'documents' in globals() else [],
        "chunks": [c.page_content for c in chunks] if 'chunks' in globals() else []
    }
