import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchProfile,
    VectorSearchAlgorithmKind,
    HnswAlgorithmConfiguration,
    SearchField
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
        embedding_dimension: int = 1536,
        cache_dir: str = ".cache"  # Add cache_dir parameter
    ):
        self.endpoint = endpoint
        self.credential = AzureKeyCredential(key)
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        self.cache_dir = cache_dir  # Store cache directory
        self.cache_expiry = timedelta(days=1)  # Set cache expiry
        
        self.index_client = SearchIndexClient(endpoint=endpoint, credential=self.credential)
        self.search_client = SearchClient(endpoint=endpoint, credential=self.credential, index_name=index_name)
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def create_index_if_not_exists(self) -> None:
        """Create the search index if it doesn't exist."""
        try:
            if self.index_name in [index.name for index in self.index_client.list_indexes()]:
                logger.info(f"Index {self.index_name} already exists")
                return

            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="vector-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
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
                SearchField(name="vector_embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
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

    def _get_upload_cache_path(self) -> str:
        """Get the path to the cache file for uploaded documents."""
        return os.path.join(self.cache_dir, f"search_index_{self.index_name}_cache.pkl")

    def _generate_upload_cache_key(self, documents: List[Document], embeddings: List[List[float]]) -> str:
        """Generate a unique cache key based on documents and embeddings."""
        key_parts = []
        for doc, embedding in zip(documents, embeddings):
            content_hash = hash(doc.page_content)
            metadata_str = str(sorted(doc.metadata.items()))
            embedding_hash = hash(str(embedding))
            key_parts.append(f"{content_hash}_{metadata_str}_{embedding_hash}")
        
        import hashlib
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()

    def _load_upload_cache(self) -> Optional[Dict]:
        """Load cached upload information if available and not expired."""
        cache_path = self._get_upload_cache_path()
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if datetime.now() - cache_data['timestamp'] > self.cache_expiry:
                logger.info("Upload cache expired, will re-upload documents")
                return None
            
            logger.info(f"Loaded upload cache for {len(cache_data['keys'])} document sets")
            return cache_data
        except Exception as e:
            logger.warning(f"Error loading upload cache: {str(e)}")
            return None

    def _save_upload_cache(self, cache_key: str):
        """Save upload cache information."""
        try:
            cache_path = self._get_upload_cache_path()
            
            # Load existing cache if available
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                # Update timestamp and add new key
                cache_data['timestamp'] = datetime.now()
                if cache_key not in cache_data['keys']:
                    cache_data['keys'].append(cache_key)
            else:
                # Create new cache
                cache_data = {
                    'timestamp': datetime.now(),
                    'keys': [cache_key]
                }
            
            # Save updated cache
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Saved upload cache with {len(cache_data['keys'])} document sets")
        except Exception as e:
            logger.warning(f"Error saving upload cache: {str(e)}")

    def upload_documents(self, documents: List[Document], embeddings: List[List[float]]) -> None:
        """Upload documents and their embeddings to the search index, using cache if available."""
        try:
            # Generate cache key for this set of documents and embeddings
            cache_key = self._generate_upload_cache_key(documents, embeddings)
            
            # Check if these documents have already been uploaded
            cache_data = self._load_upload_cache()
            if cache_data and cache_key in cache_data['keys']:
                logger.info("Documents already uploaded to search index, skipping upload")
                return
            
            # If not in cache or cache expired, proceed with upload
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
            
            # Save to cache after successful upload
            self._save_upload_cache(cache_key)

        except Exception as e:
            logger.error(f"Error uploading documents: {str(e)}")
            raise

    def vector_search(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform vector search using the query embedding."""
        try:
            # Create a search options object with the vector query
            from azure.search.documents.models import VectorizedQuery
            vector_query = VectorizedQuery(vector=query_vector, k=top_k, fields="vector_embedding")
            
            # Use the vector query in the search options
            results = self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["content", "source", "page"]
            )
            return [dict(result) for result in results]
    
        except Exception as e:
            logger.error(f"Error performing vector search: {str(e)}")
            raise