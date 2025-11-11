import os, argparse, glob, sys, time
from uuid import uuid4
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from openai import OpenAI, APIStatusError, RateLimitError

# ------------ Env & Config ------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = (os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
                      .strip().strip('"').strip("'"))
OPENROUTER_BASE    = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX     = os.getenv("PINECONE_INDEX", "abhijeetsarap")
PINECONE_HOST      = os.getenv("PINECONE_HOST")

EMBED_MODEL_NAME   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")  # 768-dim
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", "150"))

if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY missing in .env"); sys.exit(1)
if not PINECONE_API_KEY or not PINECONE_HOST:
    print("ERROR: PINECONE_API_KEY or PINECONE_HOST missing in .env"); sys.exit(1)
if "/" not in OPENROUTER_MODEL or len(OPENROUTER_MODEL) < 5:
    print(f"ERROR: OPENROUTER_MODEL looks invalid: {repr(OPENROUTER_MODEL)}"); sys.exit(1)

# ------------ Clients ------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)             # serverless uses host
embedder = SentenceTransformer(EMBED_MODEL_NAME) # 768-dim
def embed_texts(texts): return embedder.encode(texts, normalize_embeddings=True).tolist()

llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE)

# ------------ Core Logic ------------
def read_pdfs(folder):
    docs = []
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append({"source": os.path.basename(path), "page": i+1, "text": text})
    return docs

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text): break
        start = max(end - overlap, end)
    return chunks

def ingest(data_dir: str):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    docs = read_pdfs(data_dir)
    if not docs:
        print("No PDFs found."); return
    to_upsert = []
    for d in docs:
        for ch in chunk_text(d["text"]):
            to_upsert.append({"text": ch, "source": d["source"], "page": d["page"]})
    print(f"Chunks to upsert: {len(to_upsert)}")
    vecs = embed_texts([x["text"] for x in to_upsert])
    items = [{
        "id": str(uuid4()),
        "values": v,
        "metadata": {"text": x["text"], "source": x["source"], "page": x["page"]}
    } for x, v in zip(to_upsert, vecs)]
    for i in range(0, len(items), 100):
        index.upsert(vectors=items[i:i+100])
    print(f"Upserted {len(items)} vectors to '{PINECONE_INDEX}'.")
    sources = {x['metadata']['source'] for x in items}
    print(f"✅ Ingested files: {', '.join(sources)}")


SYSTEM = ("You are a precise RAG assistant. Use ONLY the provided context. "
          "If the answer isn't in the context, say you don't know. "
          "Cite sources as (source:page). Keep it concise.")

FREE_FALLBACKS = [
    "qwen/qwen2.5-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "deepseek/deepseek-r1:free",
]

def chat_with_model(messages, primary_model, debug=False):
    order = [primary_model] + [m for m in FREE_FALLBACKS if m != primary_model]
    for m in order:
        try:
            if debug: print(f"[DEBUG] Trying model: {m}")
            return llm.chat.completions.create(model=m, messages=messages, temperature=0.2)
        except (APIStatusError, RateLimitError) as e:
            if debug: print(f"[DEBUG] Rate/Provider error on {m}: {e}; rotating...")
            time.sleep(1.0)
            continue
    raise RuntimeError("All free models are currently rate-limited or unavailable.")

def build_context(question: str, k: int):
    qvec = embed_texts([question])[0]
    res = index.query(vector=qvec, top_k=k, include_metadata=True)
    blocks = []
    for m in res.matches:
        meta = m.metadata or {}
        src, pg, txt = meta.get("source","unknown"), meta.get("page",0), meta.get("text","")
        if txt: blocks.append(f"[{src}:{pg}]\n{txt}")
    return "\n\n---\n\n".join(blocks)

def ask(question: str, k: int = 5, debug: bool = False) -> str:
    context = build_context(question, k)
    if debug:
        print("\n[DEBUG] --- Retrieved Context Start ---")
        print(context[:2000] if context else "(empty)")
        print("[DEBUG] --- Retrieved Context End ---\n")
    if not context.strip():
        return "No relevant context found. Re-ingest your PDFs or use OCR for scanned PDFs."

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
    ]
    try:
        chat = chat_with_model(messages, OPENROUTER_MODEL, debug=debug)
        content = (chat.choices[0].message.content or "").strip()
        return content if content else "LLM returned empty content. Try again or switch model."
    except Exception as e:
        return f"LLM call failed: {e}"

# ------------ Optional API ------------
def run_api(port: int):
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
    app = FastAPI(title="AgentOne RAG API")
    class AskBody(BaseModel):
        question: str
        k: int = 5
        debug: bool = False
    @app.post("/ask")
    def _ask(b: AskBody):
        return {"answer": ask(b.question, b.k, b.debug)}
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

# ------------ CLI + Interactive Menu ------------
def parse_args_or_menu():
    # If called with args, use argparse. If not, show menu.
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="AgentOne RAG: ingest | ask | api")
        sub = parser.add_subparsers(dest="cmd", required=True)

        p1 = sub.add_parser("ingest"); p1.add_argument("--data", default="./data")
        p2 = sub.add_parser("ask");    p2.add_argument("--q", required=True); p2.add_argument("--k", type=int, default=5); p2.add_argument("--debug", action="store_true")
        p3 = sub.add_parser("api");    p3.add_argument("--port", type=int, default=8000)
        return parser.parse_args()

    # Interactive menu mode
    print("\n=== AgentOne RAG ===")
    print("1) Ingest PDFs")
    print("2) Ask a question")
    print("3) Run API")
    choice = input("Select [1-3]: ").strip()

    class A: pass
    a = A()

    if choice == "1":
        a.cmd = "ingest"
        a.data = input("Docs folder path [./data]: ").strip() or "./data"
    elif choice == "2":
        a.cmd = "ask"
        a.q = input("Your question: ").strip()
        k = input("Top-K (default 5): ").strip()
        a.k = int(k) if k.isdigit() else 5
        dbg = input("Enable debug? (y/N): ").strip().lower()
        a.debug = dbg in ("y","yes","1","true")
    elif choice == "3":
        a.cmd = "api"
        port = input("Port (default 8000): ").strip()
        a.port = int(port) if port.isdigit() else 8000
    else:
        print("Invalid choice."); sys.exit(2)
    return a

if __name__ == "__main__":
    a = parse_args_or_menu()
    if a.cmd == "ingest":
        ingest(getattr(a, "data", "./data"))
    elif a.cmd == "ask":
        print("\n=== Answer ===\n" + ask(a.q, getattr(a, "k", 5), debug=getattr(a, "debug", False)))
    elif a.cmd == "api":
        run_api(getattr(a, "port", 8000))
