import bs4
import requests
import os
import xml.etree.ElementTree as ET
from markdownify import MarkdownConverter
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import DocusaurusLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from crawl import *
from chroma import *

load_dotenv()


# Time to retrieve!
def retrieve(vectorstore, query):
    # Retrieve and generate using the relevant snippets of the doc
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    retrieved_docs = rag_chain_with_source.invoke(query)
    
    print(retrieved_docs)
    print('-------')
    print('Question:', retrieved_docs['question'])
    print('Answer:', retrieved_docs['answer'])
    print('Sources:')
    for doc in retrieved_docs['context']:
        print(doc.metadata['source'])



if __name__ == "__main__":
    vectorstore = load_vectorstore_from_chroma('vectordb', 'katalon_docs')
    query = "how many parallel sessions do i have for testcloud"
    retrieve(vectorstore, query)
