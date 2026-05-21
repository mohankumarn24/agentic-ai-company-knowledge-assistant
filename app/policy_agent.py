import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

# ----------------------------
# MCP + Agent Configuration
# ----------------------------
DEFAULT_MCP_URL = "http://localhost:8000/mcp"

SYSTEM_PROMPT = """You are an HR Expense Compliance Agent.
Process each claim in these steps:
1) First, use the rag_ask tool to get the company expense policy by passing questions and category "policies"
2) Evaluate the claim strictly against the retrieved policy
3) Based on your evaluation:
   - If compliant: use the approve tool
   - If non-compliant: use the reject tool
4) Then provide your decision in JSON format
You MUST use the appropriate tool (approve or reject) based on your evaluation.
"""

ASK_TEMPLATE = """Given the company expense policy, decide the outcome for this claim.

Return strictly JSON:
{{
  "decision": "approve" | "reject",
  "reason": "<one-sentence reason appropriate for the policy>",
  "violated_clause": "<optional: cite the specific clause if rejecting in detail>"
}}

Claim:
•⁠  ⁠claim_id: {claim_id}
•⁠  ⁠date: {date}
•⁠  ⁠category: {category}
•⁠  ⁠description: {description}
•⁠  ⁠amount: {amount} {currency}
•⁠  ⁠receipt_available: {receipt_available}
•⁠  ⁠pre_approved: {pre_approved}
"""

# ----------------------------
# Helpers
# ----------------------------
def load_claims_from_bytes(file_bytes: bytes) -> Dict[str, Any]:
    return json.loads(file_bytes.decode("utf-8"))

async def build_agent(mcp_url: str, model: str = "gpt-4o-mini", temperature: float = 0.0):
    #TODO: Added code
    client = MultiServerMCPClient(
        {
            "tools":
            {
                "url":mcp_url,
                "transport":"streamable_http"
            }
        }
    )
    tools = await client.get_tools()
    llm = ChatOpenAI(model=model, temperature=temperature)
    agent = create_agent(llm, tools)
    return agent

async def process_claims(agent, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    employee = data.get("employee", {})
    claims: List[Dict[str, Any]] = data.get("claims", [])

    results = []
    employee_ctx = f"{employee.get('department','')}, {employee.get('designation','')}, {employee.get('location','')}"

    for claim in claims:
        # Build the user content (policy evaluation prompt)
        user_content = ASK_TEMPLATE.format(
            employee_id=employee.get("id"),
            claim_id=claim.get("claim_id"),
            date=claim.get("date"),
            category=claim.get("category"),
            description=claim.get("description"),
            amount=claim.get("amount"),
            currency=claim.get("currency"),
            receipt_available=claim.get("receipt_available"),
            pre_approved=claim.get("pre_approved"),
            employee_context=employee_ctx,
        )

        # Pass both system and user messages so the agent knows to approve/reject after reading policy
        response = await agent.ainvoke({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        })

        try:
            final_msg = response["messages"][-1].content
        except Exception:
            final_msg = str(response)

        results.append({
            "claim_id": claim.get("claim_id"),
            "category": claim.get("category"),
            "amount": f"{claim.get('amount')} {claim.get('currency')}",
            "decision_trace": final_msg,
        })

    return results

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="HR Expense Agent (MCP)", page_icon="💼", layout="wide")
st.title("💼 HR Expense Compliance Agent (MCP + RAG)")

with st.sidebar:
    st.header("Settings")
    mcp_url = st.text_input("MCP Server URL", value=DEFAULT_MCP_URL)
    model = st.text_input("OpenAI Model", value="gpt-4o-mini")
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1)
    st.caption("MCP server should expose tools: rag_ask, approve, reject.")

uploaded = st.file_uploader("Upload claims JSON", type=["json"])

col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("Run Agent on Claims", type="primary", use_container_width=True)
with col2:
    st.write("")

if run_btn:
    if not uploaded:
        st.error("Please upload a claims JSON file first.")
        st.stop()

    try:
        data = load_claims_from_bytes(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to parse JSON: {e}")
        st.stop()

    employee = data.get("employee", {})
    st.subheader("Employee")
    st.json(employee)

    with st.spinner("Connecting to MCP server and building agent..."):
        try:
            agent = asyncio.run(build_agent(mcp_url=mcp_url, model=model, temperature=temperature))
        except Exception as e:
            st.error(f"Failed to connect/build agent: {e}")
            st.stop()

    st.info(f"Found {len(data.get('claims', []))} claim(s). Running decisions + actions...")
    with st.spinner("Evaluating claims and taking actions..."):
        try:
            results = asyncio.run(process_claims(agent, data))
        except Exception as e:
            st.error(f"Agent run failed: {e}")
            st.stop()

    st.success("Done!")
    st.subheader("Results")
    for r in results:
        with st.expander(f"Claim {r['claim_id']} — {r['category']} — {r['amount']}", expanded=False):
            st.markdown("Decision / Action Trace")
            st.write(r["decision_trace"])

    st.caption(
        "The agent evaluates each claim against policy, then finalizes it as approved or rejected. "
        "Your MCP server performs the actual actions via its tools."
    )
    
    
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