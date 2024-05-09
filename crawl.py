import bs4
import requests
import os
import xml.etree.ElementTree as ET
import html2markdown
import langchain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import DocusaurusLoader
from langchain_core.documents.base import Document
from markdownify import MarkdownConverter

# Scrape the page and get content of the html
def get_doc_content_from_url(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("theme-doc-markdown markdown")
            )
        ),
    )
    docs = loader.load()
    return docs

# Scrape the page and convert to markdown for more semantics
def get_markdown_doc_from_url(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = bs4.BeautifulSoup(
            response.text, "html.parser", parse_only=bs4.SoupStrainer(class_=("theme-doc-markdown markdown"))
        )
        # print(soup)
        converted = MarkdownConverter(strip=['img']).convert_soup(soup)
        # print(converted)
        doc = Document(
            page_content=converted,
            metadata={'source': url}
        )
        return [doc]
    else:
        print("Failed to retrieve the webpage")

# Get all urls available in the sitemap file
def get_urls_from_sitemap(url, filters):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            urls = []
            for elem in root.iter('{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc_elem = elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None and loc_elem.text.startswith(tuple(filters)):
                    urls.append(loc_elem.text)
            return urls
        else:
            print("Failed to retrieve sitemap:", response.status_code)
            return []
    except Exception as e:
        print("An error occurred:", e)
        return []

# Do retrieval from db
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
    print(get_markdown_doc_from_url('https://docs.katalon.com/katalon-studio/get-started/quick-guide-for-testers')[0].page_content)
    # print(get_urls_from_sitemap('https://docs.katalon.com/sitemap.xml', ['https://docs.katalon.com/katalon-studio', 'https://docs.katalon.com/katalon-platform']))