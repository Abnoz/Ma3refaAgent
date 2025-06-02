import os
import logging
from dotenv import load_dotenv
from colorama import Fore, Style, init
from utils.blob_storage import BlobStorageHelper
from utils.text_processor import TextProcessor
from utils.search_helper import SearchHelper
from utils.qa_chain import QAChainHelper

init()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Load and validate environment variables."""
    load_dotenv()
    
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

# In the process_documents function, update the SearchHelper initialization
def process_documents(container_name: str):
    """Process documents from blob storage and index them."""
    try:

        cache_dir = os.getenv('CACHE_DIR', '.cache')
        blob_helper = BlobStorageHelper(
            os.getenv('AZURE_STORAGE_CONNECTION_STRING'),
            cache_dir=cache_dir
        )
        text_processor = TextProcessor(cache_dir=cache_dir)
        search_helper = SearchHelper(
            endpoint=os.getenv('AZURE_SEARCH_ENDPOINT'),
            key=os.getenv('AZURE_SEARCH_KEY'),
            index_name=os.getenv('AZURE_SEARCH_INDEX_NAME'),
            cache_dir=cache_dir  # Pass cache_dir to SearchHelper
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


        logger.info("Downloading PDFs from blob storage...")
        pdf_files = blob_helper.download_pdfs_from_container(container_name)
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in container {container_name}")

        logger.info("Processing PDFs and splitting into chunks...")
        chunks = text_processor.process_pdfs(pdf_files)

        logger.info("Creating search index...")
        search_helper.create_index_if_not_exists()

        logger.info("Generating embeddings...")
        embeddings = qa_helper.generate_embeddings(chunks)

        logger.info("Uploading documents to search index...")
        search_helper.upload_documents(chunks, embeddings)

        return qa_helper

    except Exception as e:
        logger.error(f"Error in document processing: {str(e)}")
        raise

def interactive_qa(qa_helper: QAChainHelper):
    """Run interactive QA session."""
    print(f"\n{Fore.GREEN}=== Ma3refa Agent Interactive QA ==={Style.RESET_ALL}")
    print(f"{Fore.CYAN}Type your questions in Arabic. Type 'exit' to quit.{Style.RESET_ALL}\n")

    while True:
        try:
            question = input(f"{Fore.YELLOW}Question: {Style.RESET_ALL}")
            
            if question.lower() == 'exit':
                break
            
            if not question.strip():
                continue

            print(f"\n{Fore.CYAN}Searching for answer...{Style.RESET_ALL}")
            
            result = qa_helper.answer_question(question)
            
            print(f"\n{Fore.GREEN}Answer:{Style.RESET_ALL}")
            print(result['answer'])
            
            print(f"\n{Fore.BLUE}Sources:{Style.RESET_ALL}")
            for doc in result['source_documents']:
                print(f"- File: {doc['source']}, Page: {doc['page']}")
            
            print("\n" + "="*50 + "\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error during QA: {str(e)}")
            print(f"\n{Fore.RED}An error occurred. Please try again.{Style.RESET_ALL}\n")

def main():
    try:
        # Load environment variables
        setup_environment()
        
        # Get container name from environment or use default
        container_name = os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'documents')
        
        # Process documents and get QA helper
        qa_helper = process_documents(container_name)
        
        # Start interactive QA
        interactive_qa(qa_helper)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"\n{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        return 1

if __name__ == "__main__":
    main()