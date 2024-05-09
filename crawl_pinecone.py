from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from crawl import *
import os

from dotenv import load_dotenv
import requests
load_dotenv()

### Pinecone vectorstore setup 

if os.environ.get("PINECONE_API_KEY", None) is None:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

if os.environ.get("PINECONE_ENVIRONMENT", None) is None:
    raise Exception("Missing `PINECONE_ENVIRONMENT` environment variable.")

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX")

# Load
def load_docs_into_pinecone(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = PineconeVectorStore.from_documents(
        documents=splits, embedding=OpenAIEmbeddings(), index_name=PINECONE_INDEX_NAME)

if __name__ == "__main__":
    # Crawl the documents and load into Pinecone
    urls = get_urls_from_sitemap('https://docs.katalon.com/sitemap.xml', ['https://docs.katalon.com/katalon-studio', 'https://docs.katalon.com/katalon-platform'])
    docs = []
    print('Crawling')
    for url in urls:
        print(url)
        docs += get_markdown_doc_from_url(url)
    load_docs_into_pinecone(docs)