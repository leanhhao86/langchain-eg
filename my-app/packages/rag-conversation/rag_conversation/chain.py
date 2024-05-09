import os
from operator import itemgetter
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from langchain.memory import ConversationBufferMemory, DynamoDBChatMessageHistory

import boto3

# Get the service resource.
dynamodb = boto3.resource("dynamodb")



from dotenv import load_dotenv
load_dotenv()

### Pinecone vectorstore setup 

from langchain_pinecone import PineconeVectorStore

if os.environ.get("PINECONE_API_KEY", None) is None:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

if os.environ.get("PINECONE_ENVIRONMENT", None) is None:
    raise Exception("Missing `PINECONE_ENVIRONMENT` environment variable.")

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "langchain-test")

## Ingest code - you may need to run this the first time
# Load
# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# data = loader.load()

# Split
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# all_splits = text_splitter.split_documents(data)

# Add to vectorDB
# vectorstore = PineconeVectorStore.from_documents(
#     documents=all_splits, embedding=OpenAIEmbeddings(), index_name=PINECONE_INDEX_NAME
# )
# retriever = vectorstore.as_retriever()

vectorstore = PineconeVectorStore.from_existing_index(
    PINECONE_INDEX_NAME, OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

### Chromadb setup

# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings,
# )

# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = Chroma('katalon_docs', embedding_function=embedding_function, persist_directory='/Users/haole/Projects/langchain-eg/my-app/vectordb')
# retriever = vectorstore.as_retriever()

# Condense a chat history and follow-up question into a standalone question
# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""  # noqa: E501
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# RAG answer synthesis prompt
template = """Answer the question based only on the following context, return links, if there's any:
<context>
{context}
</context>"""
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        # MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ]
)

# Conversational Retrieval Chain
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


# def _combine_documents(
#     docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
# ):
#     doc_strings = [format_document(doc, document_prompt) for doc in docs]
#     return document_separator.join(doc_strings)


# def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
#     buffer = []
#     for human, ai in chat_history:
#         buffer.append(HumanMessage(content=human))
#         buffer.append(AIMessage(content=ai))
#     return buffer


# User input
class ChatHistory(BaseModel):
    question: str


_search_query = RunnableLambda(itemgetter("question"))

_inputs = RunnableParallel(
    {
        "question": lambda x: x["question"],
        "context": _search_query | retriever,
    }
)

chain = _inputs | ANSWER_PROMPT | ChatOpenAI() | StrOutputParser()


if __name__ == "__main__":
    print(chain.invoke({"question": "biometric authenticate"}))