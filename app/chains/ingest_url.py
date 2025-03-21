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
    
    links = " | ".join([a['href'] for a in soup.find_all('a', href=True)])
    
    doc = Document(
        page_content=text_content,
        metadata={
            "source": url,
            "links": links,
            "scraped_at": datetime.datetime.now().isoformat()
        }
    )
    
    web_vectorstore = await asyncio.to_thread(
        Zilliz,
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