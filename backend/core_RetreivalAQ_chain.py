# Augmented Prompt (using the RetrievalQA Chain which was DEPRECATED)

import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeLangChain

sys.path.append(os.getcwd()) # This line is necessary to import INDEX_NAME from constants (also, the directory of LangChain_Course_2_Documentation_Helper/LangChain_Course_2_Documentation_Helper is necessary)

from constants import INDEX_NAME

from typing import Any

pc = Pinecone()


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    docsearch = PineconeLangChain.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True
    )
    result = qa_chain.invoke(input={"query": query})
    return result

if __name__ == "__main__": 
    load_dotenv(find_dotenv(), override=True)
    response=run_llm(query="What is the RetrievalQA chain?")
    print(response)
    print("-"*100) 
    print(response["result"])
    print("-"*100) 
    print(response["source_documents"])