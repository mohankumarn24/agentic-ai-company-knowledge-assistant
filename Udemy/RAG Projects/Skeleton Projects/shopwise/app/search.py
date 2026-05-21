import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from sqlalchemy import select

from langchain_core.globals import set_llm_cache
from langchain_redis import RedisSemanticCache
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever

from .database import get_vector_store
from .models import Product

SYSTEM = """You are a professional product search assistant.
Help customers find the best products based on their needs.
Use the provided product catalog to recommend relevant items.
If you find matching products, provide detailed recommendations with key features.
If no products match the query, politely suggest browsing our catalog or refining the search.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",
     "Customer Query:\n{input}\n\n"
     "Product Catalog:\n{context}\n\n"
     "Please provide professional product recommendations with key features and benefits.")
])

REDIS_URL = os.getenv("REDIS_URL")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#TODO - Implement Caching



async def _build_chain():
    #TODO
    #GET Store

    #GET Retriever

    #llm
    llm = ChatOpenAI(model="gpt-4o-mini")

    #Create Stuff Documents Chain using llm and PROMPT

    #Create Retrieval Chain (rag_chain) using retriever and doc_chain

    #Return rag_chain

    

async def search_products_async(question: str):
    chain = await _build_chain()
    result = await chain.ainvoke({"input": question})
    answer: str = result["answer"]
    
    docs = result["context"]
    
    # Always try to show products if they exist in context
    # Only return empty if truly no relevant products
    
    # Extract product IDs and fetch products
    product_ids = [
        doc.metadata.get("product_id")
        for doc in docs
        if doc.metadata.get("product_id")
    ]
    
    products = []
    if product_ids:
        from .database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Product).where(Product.id.in_(product_ids)))
            db_products = result.scalars().all()
            
            products = [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "price": float(p.price),
                    "category": p.category,
                    "image_url": p.image_url,
                }
                for p in db_products
            ]
    
    contexts = [d.page_content for d in docs]
    
    return answer, products, contexts