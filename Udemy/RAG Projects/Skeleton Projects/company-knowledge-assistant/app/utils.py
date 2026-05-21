import os
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore

PG_CONN_STR = os.getenv("DATABASE_URL")




async def get_vector_store()->AsyncPGVectorStore:
    pass

