import logging
import os
import pickle
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from tenacity import retry, wait_exponential, stop_after_attempt

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
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Get top 3 most relevant results with higher similarity threshold
            results = self.search_helper.vector_search(
                query_embedding, 
                top_k=3  
            )
            
            # Convert to Documents
            documents = [
                Document(
                    page_content=result["content"],
                    metadata={
                        "source": result.get("source", "unknown"),
                        "page": result.get("page", 0),
                        "score": result.get("score", 0.0)  # Include similarity score
                    }
                )
                for result in results
            ]
            
            # Sort by score and take top 3
            documents.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
            return documents[:3]
            
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

        prompt_template = """Use the following pieces of context to answer the user's question:

Respond in the same language used in the user's question (Arabic or English).

Provide a detailed and structured answer whenever possible.

If the user asks for a summary (or uses words like "summarize", "outline", "brief", "نُبذة", "ملخّص") — respond using clear bullet points.

If the question is unclear or lacks detail, politely suggest that the user can provide more context or clarify the question for a more accurate response.

After answering, ask the user if they would like the answer summarized or if they want to add more details for better precision.


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

        self.batch_size = 20  # Process 20 documents at a time
        self.delay_between_batches = 1  # 1 second delay between batches

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
        cache_path = self._get_embeddings_cache_path(cache_key)
        if not os.path.exists(cache_path):
            logger.warning(f"Embeddings cache file not found: {cache_path}")
            return None
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            logger.info(
                f"Loaded embeddings for {len(cache_data['embeddings'])} documents from cache")
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

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
    def _get_embeddings_with_retry(self, texts):
        """Get embeddings with retry logic"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            if "429" in str(e):
                time.sleep(2)  # Additional delay on rate limit
            raise

    def get_embeddings_for_documents(self, documents: List[Document]) -> List[List[float]]:
        """Get embeddings for documents with batching and rate limiting"""
        all_embeddings = []
        texts = [doc.page_content for doc in documents]
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                batch_embeddings = self._get_embeddings_with_retry(batch)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Processed embeddings batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
                
                # Add delay between batches
                if i + self.batch_size < len(texts):
                    time.sleep(self.delay_between_batches)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//self.batch_size + 1}: {str(e)}")
                raise

        return all_embeddings

    def generate_embeddings(self, documents: List[Document]) -> List[List[float]]:
        try:
            cache_key = self._generate_cache_key(documents)
            cached_embeddings = self._load_embeddings_cache(cache_key)
            if cached_embeddings:
                return cached_embeddings
    
            texts = [doc.page_content for doc in documents]
            embeddings = self.get_embeddings_for_documents(documents)
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
