from crawl import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

# Split the docs into chunks and store in vector db
def load_docs_into_chroma(persist_directory, collection, docs):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma, reset the collection
    db = Chroma(collection, embedding_function, persist_directory=persist_directory)
    db.delete_collection()
    db = db.from_documents(splits, embedding_function, collection_name=collection, persist_directory=persist_directory)
    return db

# Load Chroma db
def load_vectorstore_from_chroma(persist_directory, collection):

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(collection, embedding_function=embedding_function, persist_directory=persist_directory)
    return vectorstore

if __name__ == "__main__":
    # Crawl the documents and load into Chroma
    urls = get_urls_from_sitemap('https://docs.katalon.com/sitemap.xml', ['https://docs.katalon.com/katalon-studio', 'https://docs.katalon.com/katalon-platform'])
    docs = []
    for url in urls[:500]:
        docs += get_markdown_doc_from_url(url)
    load_docs_into_chroma('vectordb', 'katalon_docs', docs)