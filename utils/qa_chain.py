import logging
from typing import List, Dict, Any
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
            results = self.search_helper.vector_search(query_embedding, top_k=3)
            documents = []
            for result in results:
                doc = Document(
                    page_content=result["content"],
                    metadata={
                        "source": result["source"],
                        "page": result["page"]
                    }
                )
                documents.append(doc)
            
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
        search_helper
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
            deployment=chat_deployment,
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

    def generate_embeddings(self, documents: List[Document]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        try:
            texts = [doc.page_content for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the QA chain."""
        try:
            result = self.qa_chain({"query": question})
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