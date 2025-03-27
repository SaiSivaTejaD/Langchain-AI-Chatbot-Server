import asyncio
import datetime
import logging
import requests
import sys
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin
 
from langchain.schema import Document
from langchain.vectorstores import Zilliz
from langchain_openai import OpenAIEmbeddings  # Updated import
 
# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
 
from app.config import settings  # Ensure settings.py exists in app/config/
 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
async def ingest_website_data(url: str, cached_embeddings, collection_name: str = "web_documents"):
    """
    Scrape the provided URL, extract content and links, then store as a Document in Zilliz.
    """
    logger.info(f"Starting website ingestion for {url}...")
 
    response = await asyncio.to_thread(requests.get, url)
    if response.status_code != 200:
        logger.error(f"Failed to retrieve the website. Status code: {response.status_code}")
        return
 
    soup = BeautifulSoup(response.text, "html.parser")
    main_content = soup.find("div", {"id": "content"})
    text_content = main_content.get_text(separator="\n") if main_content else soup.get_text(separator="\n")
 
    links = " | ".join([urljoin(url, a['href']) for a in soup.find_all('a', href=True)])
 
    doc = Document(
        page_content=text_content,
        metadata={
            "source": url,
            "links": links,
            "scraped_at": datetime.datetime.now().isoformat()
        }
    )
 
    # Initialize Zilliz vector store
    web_vectorstore = Zilliz(
        embedding_function=cached_embeddings,
        collection_name=collection_name,
        connection_args={
            "uri": settings.ZILLIZ_URL,
            "token": settings.ZILLIZ_AUTH_TOKEN,
        },
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64}
        },
        search_params={
            "metric_type": "COSINE",
            "params": {"ef": 10}
        },
        text_field="text",
        vector_field="vector",
        auto_id=True,
        drop_old=False
    )
 
    await asyncio.to_thread(web_vectorstore.add_documents, [doc])
    logger.info("Website ingestion complete and document stored in Zilliz Cloud.")
 
# --- New: Ingest multiple websites ---
async def ingest_multiple_websites(urls: list, cached_embeddings, collection_name: str = "web_documents"):
    """
    Given a list of URLs, concurrently ingest each website's data into the specified collection.
    """
    tasks = [ingest_website_data(url, cached_embeddings, collection_name) for url in urls]
    await asyncio.gather(*tasks)
    logger.info("All website ingestion tasks completed.")
 
# Placeholder function for initializing the retrieval chain
async def initialize_retrieval_chain():
    """
    Simulate the initialization of a retrieval chain.
    Replace this with actual initialization logic as needed.
    """
    class MockChainWrapper:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Fixed embedding model
   
    return MockChainWrapper()
 
# Placeholder function for answering and storing queries
async def answer_and_store(query: str, chain_wrapper):
    """
    Simulate answering a query using the retrieval chain.
    Replace this with actual query handling logic.
    """
    logger.info(f"Processing query: {query}")
    return f"Mock answer for: {query}"
 
# --- Example Main Execution ---
if __name__ == "__main__":
    async def main():
        # Initialize the retrieval chain
        chain_wrapper = await initialize_retrieval_chain()
 
        # List of URLs to ingest
        urls_to_ingest = [
            "https://www.wichita.gov/1710/Legal-Information",
            "https://www.wichitaareasistercities.net/",
            # Add more URLs as needed...
        ]
 
        # Ingest multiple website documents into the "web_documents" collection concurrently
        await ingest_multiple_websites(urls_to_ingest, chain_wrapper.embeddings)
 
        # Answer a sample query and store the query
        answer = await answer_and_store("What is LangChain?", chain_wrapper)
        print(answer)
 
    asyncio.run(main())