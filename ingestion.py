from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders.readthedocs import (
    ReadTheDocsLoader,
)  # Overall, ReadTheDocs helps people write documentation for packages (including langchain), however this means that ReadTheDocsLoader can only be used for reading documentation and not other types of files or writing

from langchain_openai import OpenAIEmbeddings

from langchain_pinecone import PineconeVectorStore

from functions.pinecone_index import insert_or_fetch_embeddings

import sys
import os

sys.path.append(os.getcwd())

from constants import INDEX_NAME

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs(index_name: str, file_path: str):
    loader = ReadTheDocsLoader(path=file_path, encoding="utf-8")

    raw_documents = loader.load()

    # print("-"*100)
    # print(raw_documents)
    # print("-"*100)

    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        old_url = doc.metadata["source"]
        new_url = old_url.replace("documents_2\langchain-docs\\", "https://")
        new_url_2 = new_url.replace("\\", "/")
        doc.metadata.update({"source": new_url_2})

    print(f"Going to add len{documents} to Pinecone index: {index_name}")

    insert_or_fetch_embeddings(
        index_name=index_name, delete=False, have_vectors=True, chunks=documents
    )

    print("Loading to vectorstore done")

    return documents


if __name__ == "__main__":
    # insert_or_fetch_embeddings(index_name="all", delete=True)
    docs = ingest_docs(index_name=INDEX_NAME, file_path="documents_2/langchain-docs")
    print("-" * 250)
    # for doc in docs:
    #     print(doc.metadata["source"])
