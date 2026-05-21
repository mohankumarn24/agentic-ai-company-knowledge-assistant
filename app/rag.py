from typing import List, Tuple
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.docstore.document import Document

from langchain_core .globals import set_llm_cache
from langchain_redis import RedisSemanticCache

from .utils import get_vector_store

from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever




SYSTEM = """You are a grounded company knowledge assistant.
Always base answers strictly on the provided context.
If the answer isn't present, reply with "I don't know."
Respond concisely and clearly.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",
     "Question:\n{input}\n\n"
     "Context:\n{context}\n\n"
     "Rule: Prefer the most recent policy by effective date.")
])

REDIS_URL = os.getenv("REDIS_URL")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

set_llm_cache(
    RedisSemanticCache(
        redis_url=REDIS_URL,
        embeddings=embeddings,
        distance_threshold=0.05
        # distance_threshold=0.98
        # - 0.98 is too loose / too high-risk for a RAG app with different categories. 
        #   Similar-looking prompts can reuse the previous LLM answer/cache even when retrieved docs changed
        # - Also, LangChain LLM cache usually keys around the final LLM prompt, not your API route. 
        #   If two prompts are semantically close enough, RedisSemanticCache may return old answer.
        
        # Q. What are the critical DNS records required when setting up a business email domain, and what purpose does each serve?
        # A. MX (Mail Exchange): Directs email delivery...
        #    Searches guide folder and gets result
        
        # Q. What are the number of PTOs for a full time employee?
        # A. I don't know
        #    If 'distance_threshold=0.0.05', searches guide folder, but no matching result. So, returns 'I don't know'. 
        #    If 'distance_threshold=0.98', then it may return cached data. So, returns 'MX (Mail Exchange): Directs email delivery...'
    )
)

async def _build_chain(category: str = None):
    store = await get_vector_store()
    search_kwargs={"k":int(os.getenv("RETRIEVAL_K", 5))}
    if category:
        search_kwargs["filter"]={"category":category}
    retriever = store.as_retriever(search_kwargs=search_kwargs)
#     base_retriever = store.as_retriever(search_kwargs=search_kwargs)
#     compressor = CohereRerank(
#         top_n = 3,
#         model = "rerank-multilingual-v3.0"
#     )
#     retriever = ContextualCompressionRetriever(
#         base_retriever=base_retriever,
#         base_compressor=compressor
#     )     
    llm = ChatOpenAI(model="gpt-5-nano")
    doc_chain = create_stuff_documents_chain(llm, PROMPT)
    rag_chain = create_retrieval_chain(retriever, doc_chain)        # use 'base_retriever' if using CohereRerank
    return rag_chain

## Unable to Signup for Cohere re-ranking to get API key. So, use above code without re-ranking
## Using standard vector retriever for now.
# async def _build_chain():
#     store = await get_vector_store()
#     base_retriever = store.as_retriever(search_kwargs={"k":int(os.getenv("RETRIEVAL_K", 5))})
#     compressor = CohereRerank(
#         top_n = 3,
#         model = "rerank-multilingual-v3.0"
#     )
#     retriever = ContextualCompressionRetriever(
#         base_retriever=base_retriever,
#         base_compressor=compressor
#     )
#     llm = ChatOpenAI(model="gpt-5-nano")
#     doc_chain = create_stuff_documents_chain(llm, PROMPT)
#     rag_chain = create_retrieval_chain(retriever, doc_chain)
#     return rag_chain

async def answer_with_docs_async(question: str, category: str) -> Tuple[str, List[str],List[str]]:
    chain = await _build_chain(category)
    result = await chain.ainvoke({"input": question})
    answer: str = result["answer"]

    docs: List[Document] = result["context"]
    
    unique_sources = {d.metadata.get("source") for d in docs if d.metadata.get("source")}
    sources = sorted(unique_sources)

    contexts = []
    for d in docs:
        contexts.append(d.page_content)


    return answer, sources, contexts

