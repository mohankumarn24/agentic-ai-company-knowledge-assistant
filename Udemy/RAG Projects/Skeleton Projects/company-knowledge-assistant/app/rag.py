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

async def _build_chain():
    pass

async def answer_with_docs_async(question: str) -> Tuple[str, List[str]]:
    pass
