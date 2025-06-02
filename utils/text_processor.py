import logging
from typing import List, Tuple
from io import BytesIO
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def process_pdfs(self, pdf_files: List[Tuple[str, BytesIO]]) -> List[Document]:
        """
        Process PDF files and split them into chunks.
        Returns a list of LangChain Document objects.
        """
        all_chunks = []
        
        for filename, pdf_content in pdf_files:
            try:
                logger.info(f"Processing {filename}")
                
                # Create a PDF reader object
                pdf_reader = PdfReader(pdf_content)
                
                # Extract text from each page
                pages = []
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():  # Only include non-empty pages
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1
                            }
                        )
                        pages.append(doc)
                
                # Split the documents into chunks
                chunks = self.text_splitter.split_documents(pages)
                all_chunks.extend(chunks)
                
                logger.info(f"Generated {len(chunks)} chunks from {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
        
        logger.info(f"Total chunks generated: {len(all_chunks)}")
        return all_chunks 