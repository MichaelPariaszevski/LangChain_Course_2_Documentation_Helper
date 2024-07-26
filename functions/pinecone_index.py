def delete_pinecone_index(index_name="all"):
    import pinecone

    pc = pinecone.Pinecone()
    if index_name == "all":
        indexes = pc.list_indexes().names()
        print("Deleting all indexes ... ")
        for index in indexes:
            pc.delete_index(index)
        print("Deleted all Indexes")
    else:
        print(f"Deleting index {index_name} ... ", end="")
        pc.delete_index(index_name)
        print(f"Deleted index {index_name}")


def insert_or_fetch_embeddings(
    index_name: str, delete=False, have_vectors=True, chunks=None
):  # Chunks is an optional argument
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()  # Pinecone api_key is added here
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

    index_object = pc.Index(index_name)

    print(pc.list_indexes())
    print("-" * 50)
    print(pc.list_indexes().names())
    print("-" * 50)

    if index_name.lower() == "all":
        delete_pinecone_index(index_name="all")
        print(pc.list_indexes())
        return None
    elif (
        index_name.lower() != "all"
        and index_name not in pc.list_indexes().names()
        and chunks == None
        and delete != True
        and have_vectors != True
    ):
        print(f"Creating index {index_name} and embeddings ...", end="")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter"),
        )
        print(f"Created index: {index_name}. Currently empty (no embedded vectors)")
    elif (
        index_name.lower() != "all"
        and index_name in pc.list_indexes().names()
        and delete == True
    ):
        delete_pinecone_index(index_name=index_name)
        print(pc.list_indexes())
        return None

    elif (
        index_name.lower() != "all"
        and index_name in pc.list_indexes().names()
        and index_object.describe_index_stats()["total_vector_count"] > 0
        and delete != True
    ):
        print(f"Index {index_name} already exists. Loading embeddings ...", end="")
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print("Ok")
        return vector_store
    elif (
        index_name.lower() != "all"
        and index_name in pc.list_indexes().names()
        and index_object.describe_index_stats()["total_vector_count"] < 1
        and delete != True
        and chunks != None
    ):
        print(f"Index {index_name} exists but is empty, embedding {len(chunks)} chunks")
        vector_store = Pinecone.from_documents(
            chunks, embeddings, index_name=index_name
        )
        return vector_store
    elif (
        index_name.lower() != "all"
        and index_name not in pc.list_indexes().names()
        and chunks != None
        and delete != True
    ):
        print(f"Creating index {index_name} and embeddings ...", end="")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter"),
        )
        vector_store = Pinecone.from_documents(
            chunks, embeddings, index_name=index_name
        )
        print("Ok")
        return vector_store
    else:
        print("Invalid set of parameters")

    print("-" * 200)


if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(), override=True)

    from pinecone import Pinecone

    pc = Pinecone()

    index = pc.Index("langchain-doc-helper-index")

    # index_info=index.describe_index_stats()

    print(index.describe_index_stats()["total_vector_count"])
    # print("-"*100)
    # print(index)
    # print("-"*100)
    # print(type(index))
