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
            'AZURE_STORAGE_CONNECTION_STRING',
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT',
            'AZURE_OPENAI_CHAT_DEPLOYMENT',
            'AZURE_SEARCH_ENDPOINT',
            'AZURE_SEARCH_KEY',
            'AZURE_SEARCH_INDEX_NAME',
            'AZURE_OPENAI_API_VERSION'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Initialize components
        cache_dir = os.getenv('CACHE_DIR', '.cache')
        container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'documents')
        
        blob_helper = BlobStorageHelper(
            os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
            cache_dir=cache_dir
        )
        text_processor = TextProcessor(cache_dir=cache_dir)
        search_helper = SearchHelper(
            endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
            key=os.getenv('AZURE_SEARCH_KEY'),
            index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
            cache_dir=cache_dir
        )
        qa_helper = QAChainHelper(
            openai_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            embedding_deployment=os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT'),
            chat_deployment=os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT'),
            search_helper=search_helper,
            cache_dir=cache_dir
        )

        # Process documents
        logger.info("Downloading PDFs from blob storage...")
        pdf_files = blob_helper.download_pdfs_from_container(container_name)
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in container {container_name}")

        logger.info("Processing PDFs and splitting into chunks...")
        chunks = text_processor.process_pdfs(pdf_files)

        logger.info("Creating search index...")
        search_helper.create_index_if_not_exists()
        
        # Check if we have cached embeddings for these chunks
        cache_key = qa_helper._generate_cache_key(chunks)
        cached_embeddings = qa_helper._load_embeddings_cache(cache_key)
        
        if cached_embeddings:
            logger.info("Using cached embeddings...")
            embeddings = cached_embeddings
        else:
            logger.info("Generating embeddings...")
            embeddings = qa_helper.generate_embeddings(chunks)
        
        # Check if documents are already uploaded to search index
        upload_cache_key = search_helper._generate_upload_cache_key(chunks, embeddings)
        upload_cache = search_helper._load_upload_cache()
        
        if upload_cache and upload_cache_key in upload_cache.get('keys', []):
            logger.info("Documents already uploaded to search index, skipping upload")
        else:
            logger.info("Uploading documents to search index...")
            search_helper.upload_documents(chunks, embeddings)

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
@app.post("/api/ask", response_model=AnswerResponse)
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
                
                prompt_template = """Use the following pieces of context and the knowldge to answer the question at the end. 
                If you don't know the answer, Tell the user that you are not sure about what he is saying
                and ask him to provide you with more context to be able to help him be creative 
                in those time so you dont make the user feel that you are dumb , don't try to make up an answer Except for greetings.
                regardless of the language of the context or question.
                Respond in the same language used in the user’s question (Arabic or English).

                Context: {context}
                Question: {question}
                """
                
                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                # Create a real-time streaming handler
                streaming_handler = RealTimeStreamingHandler()
                
                # Configure the LLM with the streaming handler
                streaming_llm = qa_helper.llm.with_config({"callbacks": [streaming_handler]})
                
                # Create a streaming QA chain
                streaming_qa_chain = RetrievalQA.from_chain_type(
                    llm=streaming_llm,
                    chain_type="stuff",
                    retriever=qa_helper.retriever,
                    chain_type_kwargs={"prompt": PROMPT},
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
