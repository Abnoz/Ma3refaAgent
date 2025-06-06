import os
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from utils.blob_storage import BlobStorageHelper
from utils.text_processor import TextProcessor
from utils.search_helper import SearchHelper
from utils.qa_chain import QAChainHelper
from langchain.callbacks.base import BaseCallbackHandler
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Ma3refa Agent API", description="API for interacting with Ma3refa Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    content: str
    source: str
    page: int

class AnswerResponse(BaseModel):
    answer: str
    source_documents: Optional[List[SourceDocument]] = None

# Custom streaming callback handler
class StreamingResponseCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.streaming_started = False

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
        self.streaming_started = True

# Initialize QA helper
def initialize_qa_helper():
    try:
        # Setup environment variables
        required_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT',
            'AZURE_OPENAI_CHAT_DEPLOYMENT',
            'AZURE_SEARCH_ENDPOINT',
            'AZURE_SEARCH_KEY',
            'AZURE_SEARCH_INDEX_NAME',
            'AZURE_OPENAI_API_VERSION',
            'AZURE_VISION_ENDPOINT',  # Added for Azure Vision
            'AZURE_VISION_KEY'  # Added for Azure Vision
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Add necessary imports
        from azure.search.documents.indexes import SearchIndexClient
        from azure.core.credentials import AzureKeyCredential

        # Initialize components
        cache_dir = os.getenv('CACHE_DIR', '.cache')
        
        # Check if the index exists
        index_client = SearchIndexClient(
            endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
            credential=AzureKeyCredential(os.getenv('AZURE_SEARCH_KEY'))
        )
        
        index_exists = os.getenv('AZURE_SEARCH_INDEX_NAME') in [index.name for index in index_client.list_indexes()]
        if not index_exists:
            raise ValueError(f"Search index '{os.getenv('AZURE_SEARCH_INDEX_NAME')}' does not exist. Please create it first.")
        
        # Initialize search helper and QA helper without processing documents
        search_helper = SearchHelper(
            endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
            key=os.getenv('AZURE_SEARCH_KEY'),
            index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
            cache_dir=cache_dir
        )
        
        # Initialize QA helper with embeddings
        qa_helper = QAChainHelper(
            openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            embedding_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
            chat_deployment=os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT'),
            search_helper=search_helper,
            cache_dir=cache_dir
        )

        logger.info("Successfully connected to existing search index")
        return qa_helper

    except Exception as e:
        logger.error(f"Error initializing QA helper: {str(e)}")
        raise

# Initialize QA helper at startup
qa_helper = None

@app.on_event("startup")
async def startup_event():
    global qa_helper
    try:
        qa_helper = initialize_qa_helper()
        logger.info("QA helper initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize QA helper: {str(e)}")

# Standard JSON response endpoint
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if qa_helper is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    
    try:
        result = qa_helper.answer_question(request.question)
        return AnswerResponse(
            answer=result["answer"],
            source_documents=result["source_documents"]
        )
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming endpoint
@app.post("/api/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    if qa_helper is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    
    try:
        class RealTimeStreamingHandler(BaseCallbackHandler):
            def __init__(self):
                self.streaming_started = False

            async def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.streaming_started = True
                # Yield the token immediately
                yield token
        
        async def response_generator():
            try:
                from langchain.chains import RetrievalQA
                from langchain.prompts import PromptTemplate
                
                prompt_template = """Use the following pieces of context to answer the question at the end. 
                Only use the information from the provided context. If you don't find the specific information in the context:
                1. Clearly state that you cannot find the exact information in the available documents
                2. Ask the user if they would like to rephrase or provide more details to help you find relevant information
                
                Important Rules:
                - Answer ONLY based on the provided context - do not use external knowledge
                - If the question is in English, answer in English
                - If the question is in Arabic, answer in Arabic
                - If asked for a summary or overview, structure the response in bullet points
                - Maximum 2-3 references from the context
                - If the question is unclear or too broad, ask for clarification
                - After answering, ask if the user would like:
                  * A more detailed explanation
                  * A summary of the answer
                  * To rephrase their question for better results

                Context: {context}
                Question: {question}
                Answer: """
                
                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                # Create a real-time streaming handler
                streaming_handler = RealTimeStreamingHandler()
                
                # Configure the LLM with the streaming handler
                streaming_llm = qa_helper.llm.with_config({"callbacks": [streaming_handler]})
                
                # Create a streaming QA chain with limited number of documents
                streaming_qa_chain = RetrievalQA.from_chain_type(
                    llm=streaming_llm,
                    chain_type="stuff",
                    retriever=qa_helper.retriever,
                    chain_type_kwargs={
                        "prompt": PROMPT,
                        "document_prompt": PromptTemplate(
                            template="Content: {page_content}\nSource: {source}",
                            input_variables=["page_content", "source"]
                        ),
                        "document_separator": "\n\n"
                    },
                    return_source_documents=True
                )
                
                # This will stream tokens through the callback as they're generated
                await streaming_qa_chain.ainvoke({"query": request.question})
                    
            except Exception as e:
                logger.error(f"Error in streaming response: {str(e)}")
                yield f"Error: {str(e)}"
        
        # Return a streaming response
        return StreamingResponse(
            response_generator(),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Error setting up streaming response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming endpoint with sources
@app.post("/api/ask/stream-with-sources")
async def ask_question_stream_with_sources(request: QuestionRequest):
    if qa_helper is None:
        raise HTTPException(status_code=503, detail="Service not initialized yet")
    
    try:
        # First get the answer and sources using the standard method
        result = qa_helper.answer_question(request.question)
        
        # Then create a streaming response that includes both the answer and sources
        async def response_generator():
            # First yield the answer
            yield result["answer"]
            
            # Then yield a separator
            yield "\n\n---SOURCES---\n\n"
            
            # Then yield the sources in a structured format
            for doc in result["source_documents"]:
                source_text = f"Source: {doc['source']}, Page: {doc['page']}\n"
                yield source_text
        
        # Return a streaming response
        return StreamingResponse(
            response_generator(),
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Error in streaming response with sources: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    if qa_helper is None:
        return {"status": "initializing"}
    return {"status": "healthy"}

# Run the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
