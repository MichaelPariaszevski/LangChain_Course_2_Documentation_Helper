from dotenv import load_dotenv, find_dotenv 

load_dotenv(find_dotenv(), override=True)

from functions.pinecone_index import insert_or_fetch_embeddings 

insert_or_fetch_embeddings(index_name="All", delete=True, have_vectors=False, chunks=None)

insert_or_fetch_embeddings(index_name="langchain-doc-helper-index", delete=False, have_vectors=False, chunks=None)