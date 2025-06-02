import logging
from typing import List, Dict, Any
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchProfile,
    VectorSearchAlgorithmKind
)
from azure.core.credentials import AzureKeyCredential
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SearchHelper:
    def __init__(
        self,
        endpoint: str,
        key: str,
        index_name: str,
        embedding_dimension: int = 1536  
    ):
        self.endpoint = endpoint
        self.credential = AzureKeyCredential(key)
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        
        self.index_client = SearchIndexClient(endpoint=endpoint, credential=self.credential)
        self.search_client = SearchClient(endpoint=endpoint, credential=self.credential, index_name=index_name)

    def create_index_if_not_exists(self) -> None:
        """Create the search index if it doesn't exist."""
        try:
            if self.index_name in [index.name for index in self.index_client.list_indexes()]:
                logger.info(f"Index {self.index_name} already exists")
                return

            vector_search = VectorSearch(
                algorithms=[
                    VectorSearchAlgorithmConfiguration(
                        name="vector-config",
                        kind="HNSW",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="vector-config"
                    )
                ]
            )

            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="vector_embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                          searchable=True,
                          vector_search_dimensions=self.embedding_dimension,
                          vector_search_profile_name="vector-profile"),
                SimpleField(name="source", type=SearchFieldDataType.String),
                SimpleField(name="page", type=SearchFieldDataType.Int32)
            ]

            index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
            self.index_client.create_index(index)
            logger.info(f"Created index {self.index_name}")

        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise

    def upload_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Upload documents and their embeddings to the search index."""
        try:
            docs_to_upload = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                search_doc = {
                    "id": f"doc_{i}",
                    "content": doc.page_content,
                    "vector_embedding": embedding,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0)
                }
                docs_to_upload.append(search_doc)

            batch_size = 1000
            for i in range(0, len(docs_to_upload), batch_size):
                batch = docs_to_upload[i:i + batch_size]
                self.search_client.upload_documents(documents=batch)
                logger.info(f"Uploaded batch of {len(batch)} documents")

            logger.info(f"Successfully uploaded {len(docs_to_upload)} documents to the index")

        except Exception as e:
            logger.error(f"Error uploading documents: {str(e)}")
            raise

    def vector_search(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform vector search using the query embedding."""
        try:
            results = self.search_client.search(
                search_text=None,
                vector=query_vector,
                top_k=top_k,
                vector_fields="vector_embedding",
                select=["content", "source", "page"]
            )
            return [dict(result) for result in results]

        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
            raise 