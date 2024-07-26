from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

from langchain.chains.retrieval import create_retrieval_chain

from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from constants import INDEX_NAME

index_name = INDEX_NAME


def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    llm = ChatOpenAI(model_name="gpt-4o-mini")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_documents_chain = create_stuff_documents_chain(
        llm=llm, prompt=retrieval_qa_chat_prompt
    )

    qa_chain = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )

    result = qa_chain.invoke(input={"input": query})

    return result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain chain?")

    print(res)
    print("-" * 100)
    print(res["answer"])
    print("-" * 100)
    print([doc.metadata["source"] for doc in res["context"]])
