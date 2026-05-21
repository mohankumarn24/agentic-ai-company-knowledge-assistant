import asyncio, time
from pathlib import Path
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.requests import Request
from starlette.staticfiles import StaticFiles



# ---- MCP (streamable_http) ----
from mcp.server.fastmcp import FastMCP

# ---- Your RAG + Ingest imports ----
from .rag import answer_with_docs_async            # must return (answer, sources, contexts)
from .ingest import run_ingest_async               # your async ingest job

# ========== MCP SERVER ==========
mcp = FastMCP(
    name="company-kb",
    instructions="Provide tools to query company knowledge and take simple actions.",
    json_response=True,
)



# Tool schemas
class AskParams(BaseModel):
    question: str = Field(...)
    category: str | None = None

class AskResult(BaseModel):
    answer: str
    sources: List[str]
    contexts: List[str]


#TODO: Define your MCP tools here

@mcp.tool(description="Ask the RAG system by passing question and category and get answer + sources + contexts.")
async def rag_ask(params: AskParams) -> AskResult:
    answer, sources, contexts = await answer_with_docs_async(params.question, params.category)
    return AskResult(answer=answer, sources=sources, contexts=contexts)    

@mcp.tool(description="Approve a compliant expense claim and record the approval.")
async def approve() -> str:
    print("APPROVED")
    return "APPROVED"

@mcp.tool(description="Reject a non-compliant expense claim and record the rejection.")
async def reject() -> str:
    print("REJECTED")
    return "REJECTED"



#Export a Starlette app with streamable HTTP MCP endpoint at ⁠ /mcp ⁠
app = mcp.streamable_http_app()  


# ========== STATIC / UI ==========
static_dir = (Path(__file__).resolve().parent / "static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ========== INGEST STATE ==========
_ingest_lock = asyncio.Lock()
_ingest_task: asyncio.Task | None = None
_ingest_last: Dict[str, Any] = {
    "status": "idle",      # idle | running | succeeded | failed
    "started_at": None,
    "finished_at": None,
    "stats": None,         # {"documents":..., "chunks":...}
    "error": None,
}

# ========== ROUTE HANDLERS ==========
async def ui_handler(request: Request):
    """Root page -> serve your index.html if present, else a small hello."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return PlainTextResponse("Company Knowledge Assistant (MCP + Starlette)", status_code=200)

async def ask_handler(request: Request):
    """POST /ask  -> call RAG directly (same impl as before)."""
    try:
        payload = await request.json()
        question = payload.get("question") or ""
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    if not question.strip():
        return JSONResponse({"error": "Missing 'question'"}, status_code=400)

    start = time.perf_counter()
    # Optional: category to scope retrieval (adjust to your needs)
    category = payload.get("category") or "policies"

    answer, sources, contexts = await answer_with_docs_async(question, category)
    elapsed = time.perf_counter() - start
    print(f"⏱️ /ask execution took {elapsed:.2f} seconds")

    return JSONResponse({
        "answer": answer,
        "sources": sources,
        "contexts": contexts
    })

async def ingest_job():
    """Background ingest job."""
    _ingest_last.update({
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "stats": None,
        "error": None
    })
    try:
        stats = await run_ingest_async()
        _ingest_last.update({
            "status": "succeeded",
            "finished_at": time.time(),
            "stats": stats
        })
    except Exception as e:
        _ingest_last.update({
            "status": "failed",
            "finished_at": time.time(),
            "error": str(e)
        })

async def ingest_handler(request: Request):
    """POST /ingest  -> kick off ingest once."""
    global _ingest_task
    async with _ingest_lock:
        if _ingest_task and not _ingest_task.done():
            return JSONResponse({"ok": False, "message": "Ingestion already running"}, status_code=409)
        _ingest_task = asyncio.create_task(ingest_job())
    return JSONResponse({"ok": True, "message": "Ingestion started"})

async def ingest_status_handler(request: Request):
    """GET /ingest/status"""
    return JSONResponse({"ok": True, **_ingest_last})

async def mcp_health(request: Request):
    """GET /mcp/health -> optional MCP health endpoint."""
    return JSONResponse({"ok": True})



# ========== ROUTES ==========
# Root UI
app.router.add_route("/", ui_handler, methods=["GET"])
# Health for MCP mount (optional)
app.router.add_route("/mcp/health", mcp_health, methods=["GET"])
# RAG API
app.router.add_route("/ask", ask_handler, methods=["POST"])
# Ingest
app.router.add_route("/ingest", ingest_handler, methods=["POST"])
app.router.add_route("/ingest/status", ingest_status_handler, methods=["GET"])
# Test tools directly


## 1. RUN 
#     Local Terminal:
#       Run below command in local terminal
#           pip install mcp langchain_mcp_adapters streamlit langgraph
#
#     Bash:
#       cd /d/dev/github/agentic-ai-company-knowledge-assistant
#       docker compose down -v
#       docker compose up --build   -> if running first time, otherwise run 'docker compose up'
#       docker restart cka-app      -> if needed
#       
#       Launch: http://localhost:8000/
#       Ingest data
#
#       http://localhost:8000/mcp/health
#       http://localhost:8000/mcp -> you cannot run this endpoint fom browser, use Postman 
#     
#     Launch Postman > New > MCP > HTTP http://localhost:8000/mcp > Connect ? Fill fields and Run
#     
#       Question : What is the PTO policy?
#       Category : policies
#
#       Observe results



## 2. RUN
#     cd D:\dev\github\agentic-ai-company-knowledge-assistant\app
#     python -m streamlit run .\policy_agent.py
#     
#     Browse 'claim_data.json' file from 'app' folder and click 'Run Agent'
#     change below value to 'true/false' and observe results:
#       "pre_approved": true
#
#        {
#           "decision": "approve",
#           "reason": "The claim is compliant as it is pre-approved and the amount is within the allowable limit for travel expenses.",
#           "violated_clause": null
#        }


## SQL CODE:
# Connect to Postgres running on port 5432 in docker
# New connection > Host: localhost, Port: 5433, Database: postgres, Username: postgres, Password: postgres > Test connection
# 
# select
# 	langchain_metadata
# from
# 	langchain_pg_embedding;


# select
# 	indexname,
# 	indexdef
# from
# 	pg_indexes
# where
# 	tablename = 'langchain_pg_embedding'
# 	and indexname = 'hnsw_idx';


# select
# 	langchain_id,
# 	"content",
# 	embedding,
# 	langchain_metadata,
# 	category
# from
# 	public.langchain_pg_embedding;