import logging
import os
import pickle
import re
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
from io import BytesIO
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, cache_dir: str = ".cache", 
                 min_chunk_size: int = 200, max_chunk_size: int = 1000, 
                 use_adaptive_chunking: bool = True):
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.use_adaptive_chunking = use_adaptive_chunking
        
        # Initialize with default values, will be adapted per document if adaptive chunking is enabled
        self.text_splitter = self._create_text_splitter(chunk_size, chunk_overlap)
        
        self.cache_dir = cache_dir
        self.cache_expiry = timedelta(days=1)
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _create_text_splitter(self, chunk_size, chunk_overlap):
        """Create a text splitter with the given parameters."""
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    
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

    def _analyze_text_complexity(self, text: str) -> Dict:
        """Analyze text complexity to determine optimal chunk size."""
        # Calculate text statistics
        stats = {
            'length': len(text),
            'avg_sentence_length': 0,
            'avg_word_length': 0,
            'special_content_ratio': 0
        }
        
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]\s+', text)
        if sentences:
            stats['avg_sentence_length'] = sum(len(s) for s in sentences) / len(sentences)
        
        # Calculate average word length
        words = re.findall(r'\b\w+\b', text)
        if words:
            stats['avg_word_length'] = sum(len(w) for w in words) / len(words)
        
        # Check for special content (code blocks, tables, etc.)
        code_pattern = re.compile(r'(```|\{\{|def\s+\w+\(|function\s+\w+\(|class\s+\w+:|import\s+\w+)')
        table_pattern = re.compile(r'\|[-|]+\|')
        list_pattern = re.compile(r'^\s*[\*\-\d]+\.?\s+')
        
        special_content_count = (
            len(code_pattern.findall(text)) + 
            len(table_pattern.findall(text)) + 
            len(list_pattern.findall(text))
        )
        
        if len(text) > 0:
            stats['special_content_ratio'] = special_content_count / (len(text) / 100)  # per 100 chars
        
        return stats
    
    def _determine_optimal_chunk_size(self, text_stats: Dict) -> Tuple[int, int]:
        """Determine optimal chunk size based on text complexity analysis."""
        base_chunk_size = self.default_chunk_size
        base_overlap = self.default_chunk_overlap
        
        # Adjust for sentence length - longer sentences need larger chunks
        if text_stats['avg_sentence_length'] > 150:
            base_chunk_size += 200
            base_overlap += 25
        elif text_stats['avg_sentence_length'] < 50:
            base_chunk_size -= 100
        
        # Adjust for special content - code, tables need larger chunks to preserve context
        if text_stats['special_content_ratio'] > 1.0:  # Significant special content
            base_chunk_size += 300
            base_overlap += 50
        
        # Ensure we stay within bounds
        chunk_size = max(self.min_chunk_size, min(self.max_chunk_size, base_chunk_size))
        chunk_overlap = max(20, min(chunk_size // 3, base_overlap))
        
        return chunk_size, chunk_overlap

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
                
                # Process each page individually
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        # Extract text from the current page
                        text = page.extract_text()
                        
                        if not text.strip():  # Skip empty pages
                            continue
                            
                        # Create a document for this page
                        page_doc = Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1
                            }
                        )
                        
                        # If adaptive chunking is enabled, analyze this page and adjust chunk size
                        if self.use_adaptive_chunking:
                            # Analyze text complexity for this specific page
                            text_stats = self._analyze_text_complexity(text)
                            
                            # Determine optimal chunk size for this page
                            chunk_size, chunk_overlap = self._determine_optimal_chunk_size(text_stats)
                            
                            logger.info(f"Adaptive chunking for {filename} page {page_num+1}: size={chunk_size}, overlap={chunk_overlap}")
                            
                            # Create a page-specific text splitter with the optimal parameters
                            page_text_splitter = self._create_text_splitter(chunk_size, chunk_overlap)
                            page_chunks = page_text_splitter.split_documents([page_doc])
                        else:
                            # Use default text splitter
                            page_chunks = self.text_splitter.split_documents([page_doc])
                        
                        # Add the chunks from this page to our collection
                        all_chunks.extend(page_chunks)
                        
                        logger.info(f"Generated {len(page_chunks)} chunks from {filename} page {page_num+1}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {filename} page {page_num+1}: {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
        
        logger.info(f"Total chunks generated: {len(all_chunks)}")
        self._save_cache(cache_key, all_chunks)
        
        return all_chunks