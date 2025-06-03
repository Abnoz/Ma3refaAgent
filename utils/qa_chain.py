import logging
import os
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

logger = logging.getLogger(__name__)

class CustomRetriever(BaseRetriever):
    search_helper: Any
    embeddings: Any

    def __init__(self, search_helper: Any, embeddings: Any):
        super().__init__()
        self.search_helper = search_helper
        self.embeddings = embeddings

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.search_helper.vector_search(
                query_embedding, top_k=3)
            documents = [
                Document(
                    page_content=result["content"],
                    metadata={
                        "source": result.get("source", "unknown"),
                        "page": result.get("page", 0)
                    }
                )
                for result in results
            ]
            return documents
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            raise


class QAChainHelper:
    def __init__(
        self,
        openai_endpoint: str,
        openai_api_key: str,
        openai_api_version: str,
        embedding_deployment: str,
        chat_deployment: str,
        search_helper,
        cache_dir: str = ".cache"
    ):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=openai_endpoint,
            api_key=openai_api_key,
            api_version=openai_api_version,
            deployment=embedding_deployment
        )

        self.llm = AzureChatOpenAI(
            azure_endpoint=openai_endpoint,
            api_key=openai_api_key,
            api_version=openai_api_version,
            deployment_name=chat_deployment,
            temperature=0
        )

        self.retriever = CustomRetriever(search_helper, self.embeddings)

        prompt_template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Always answer in Arabic, regardless of the language of the context or question.

            Context: {context}
            Question: {question}
            Answer in Arabic:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        self.cache_dir = cache_dir

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_embeddings_cache_path(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"embeddings_{cache_key}_cache.pkl")

    def _generate_cache_key(self, documents: List[Document]) -> str:
        key_parts = []
        for doc in documents:
            content_hash = hash(doc.page_content)
            metadata_str = str(sorted(doc.metadata.items()))
            key_parts.append(f"{content_hash}_{metadata_str}")
        import hashlib
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()

    def _load_embeddings_cache(self, cache_key: str) -> Optional[List[List[float]]]:  
        # Always use the specific cache file regardless of the cache_key parameter
        cache_path = os.path.join(self.cache_dir, "embeddings_e93c742c576835c4efea9de005956062_cache.pkl")
        if not os.path.exists(cache_path):
            logger.warning(f"Specified embeddings cache file not found: {cache_path}")
            return None
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            logger.info(
                f"Loaded embeddings for {len(cache_data['embeddings'])} documents from specified cache file")
            return cache_data['embeddings']
        except Exception as e:
            logger.warning(f"Error loading embeddings cache: {str(e)}")
            return None

    def _save_embeddings_cache(self, cache_key: str, embeddings: List[List[float]]):
        try:
            cache_data = {
                # Removed timestamp
                'embeddings': embeddings
            }
            with open(self._get_embeddings_cache_path(cache_key), 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(
                f"Saved embeddings for {len(embeddings)} documents to cache")
        except Exception as e:
            logger.warning(f"Error saving embeddings cache: {str(e)}")

    def generate_embeddings(self, documents: List[Document]) -> List[List[float]]:
        try:
            cache_key = self._generate_cache_key(documents)
            cached_embeddings = self._load_embeddings_cache(cache_key)
            if cached_embeddings:
                return cached_embeddings
    
            texts = [doc.page_content for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)
            self._save_embeddings_cache(cache_key, embeddings)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def answer_question(self, question: str) -> Dict[str, Any]:
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", 0)
                    }
                    for doc in result["source_documents"]
                ]
            }
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise
