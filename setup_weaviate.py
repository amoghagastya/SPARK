import os

import weaviate
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Weaviate


def setup_weaviate():
    client = weaviate.Client(
        url='https://spark-2l75d3ky.weaviate.network', auth_client_secret=weaviate.AuthApiKey("")
    )
    # clear this class first
    client.schema.delete_class("Spark")
    class_definition = {
        "class": "Spark",
        "vectorIndexConfig": {
            "distance": "cosine" # Set to "cosine" for English models; "dot" for multilingual models
        }
    }
    client.schema.create_class(class_definition)
    
    loader = CSVLoader(file_path="prompts.csv")
    data = loader.load()
    embeddings = CohereEmbeddings(cohere_api_key=os.environ['COHERE_API_KEY'], model="embed-english-light-v3.0")
    vectorstore = Weaviate.from_documents(
            data, embeddings, client=client, by_text=False,
            index_name="Spark"
        )
    return vectorstore.as_retriever(k=2)