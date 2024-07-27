from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

import sys
import os

from langchain.chains.retrieval import create_retrieval_chain

from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.chains.history_aware_retriever import create_history_aware_retriever

sys.path.append(
    os.getcwd()
)  # This line is necessary to import INDEX_NAME from constants (also, the directory of LangChain_Course_2_Documentation_Helper/LangChain_Course_2_Documentation_Helper is necessary)

from constants import INDEX_NAME

from typing import List, Any, Dict

index_name = INDEX_NAME


def run_llm_with_memory(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    llm = ChatOpenAI(model_name="gpt-4o-mini")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(
        llm=llm, prompt=retrieval_qa_chat_prompt
    )

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa_chain = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa_chain.invoke(input={"input": query, "chat_history": chat_history})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }

    # new_result["source_documents"]=[doc.metadata["source"].replace("documents\\", "") for doc in new_result["source_documents"]]

    return new_result


if __name__ == "__main__":
    res = run_llm_with_memory(query="What is a LangChain chain?")

    print(res)
    print("-" * 100)
    print(res["result"])
    print("-" * 100)
    print(res["source_documents"])
    print("-" * 100)
    print([doc.metadata["source"] for doc in res["source_documents"]])
