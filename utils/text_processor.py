import logging
import os
import pickle
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
from io import BytesIO
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, cache_dir: str = ".cache"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        self.cache_dir = cache_dir
        self.cache_expiry = timedelta(days=1)
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the path to the cache file for processed chunks."""
        return os.path.join(self.cache_dir, f"chunks_{cache_key}_cache.pkl")
    
    def _generate_cache_key(self, pdf_files: List[Tuple[str, BytesIO]]) -> str:
        """Generate a unique cache key based on filenames and modification times."""
        # Use filenames and their sizes as a simple cache key
        key_parts = []
        for filename, content in pdf_files:
            size = len(content.getvalue())
            key_parts.append(f"{filename}_{size}")
        
        # Join and hash to create a shorter key
        import hashlib
        return hashlib.md5('_'.join(key_parts).encode()).hexdigest()
    
    def _load_cache(self, cache_key: str) -> Optional[List[Document]]:
        """Load cached chunks if available and not expired."""
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if datetime.now() - cache_data['timestamp'] > self.cache_expiry:
                logger.info("Chunks cache expired, will reprocess PDFs")
                return None
            
            logger.info(f"Loaded {len(cache_data['chunks'])} chunks from cache")
            return cache_data['chunks']
        except Exception as e:
            logger.warning(f"Error loading chunks cache: {str(e)}")
            return None
    
    def _save_cache(self, cache_key: str, chunks: List[Document]):
        """Save processed chunks to cache."""
        try:
            cache_data = {
                'timestamp': datetime.now(),
                'chunks': chunks
            }
            
            with open(self._get_cache_path(cache_key), 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Saved {len(chunks)} chunks to cache")
        except Exception as e:
            logger.warning(f"Error saving chunks cache: {str(e)}")

    def process_pdfs(self, pdf_files: List[Tuple[str, BytesIO]]) -> List[Document]:
        """Process PDF files and split them into chunks, using cache if available."""
        # Generate a cache key based on the PDF files
        cache_key = self._generate_cache_key(pdf_files)
        
        # Try to load from cache first
        cached_chunks = self._load_cache(cache_key)
        if cached_chunks:
            return cached_chunks
        
        # If no cache or expired, process the PDFs
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
        
        # Save to cache for future use
        self._save_cache(cache_key, all_chunks)
        
        return all_chunks