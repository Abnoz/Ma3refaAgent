import os
import logging
import pickle
from typing import List, Tuple, Dict, Any
from io import BytesIO

logger = logging.getLogger(__name__)

class LocalFileHelper:
    def __init__(self, knowledge_dir: str = "Knowledge"):
        self.knowledge_dir = knowledge_dir
        if not os.path.exists(knowledge_dir):
            os.makedirs(knowledge_dir)

    def read_pdfs_from_directory(self) -> List[Tuple[str, BytesIO]]:
        """Read all PDF files from the Knowledge directory and its subdirectories."""
        pdf_files = []
        try:
            # Walk through all subdirectories
            for root, _, files in os.walk(self.knowledge_dir):
                # Get relative path from Knowledge directory
                rel_path = os.path.relpath(root, self.knowledge_dir)
                
                # Filter PDF files
                pdf_filenames = [f for f in files if f.lower().endswith('.pdf')]
                
                for filename in pdf_filenames:
                    file_path = os.path.join(root, filename)
                    try:
                        # Create a relative path for the file that includes its subdirectory
                        relative_filename = os.path.join(rel_path, filename) if rel_path != '.' else filename
                        
                        # Read the PDF file into BytesIO
                        with open(file_path, 'rb') as f:
                            pdf_content = BytesIO(f.read())
                        pdf_files.append((relative_filename, pdf_content))
                        logger.info(f"Successfully read PDF file: {relative_filename}")
                    except Exception as e:
                        logger.error(f"Error reading PDF file {file_path}: {str(e)}")
                        continue
            
            logger.info(f"Successfully read {len(pdf_files)} PDF files from {self.knowledge_dir} and its subdirectories")
            return pdf_files
        except Exception as e:
            logger.error(f"Error reading PDF files from directory: {str(e)}")
            raise 

    def read_cached_raw_files(self, cache_dir: str = ".cache") -> List[Dict[str, Any]]:
        """Read all cached raw files from the .cache directory.
        
        Args:
            cache_dir (str): Path to the cache directory. Defaults to ".cache".
            
        Returns:
            List[Dict[str, Any]]: List of page data dictionaries containing text content and metadata.
        """
        all_pages_data = []
        try:
            # Check if cache directory exists
            if not os.path.exists(cache_dir):
                logger.warning(f"Cache directory {cache_dir} does not exist")
                return all_pages_data

            # Read all pickle files in the cache directory
            for filename in os.listdir(cache_dir):
                if filename.startswith("raw_results_") and filename.endswith(".pkl"):
                    cache_path = os.path.join(cache_dir, filename)
                    try:
                        with open(cache_path, 'rb') as f:
                            cache_data = pickle.load(f)
                        
                        # Each cache file contains page data
                        if 'pages' in cache_data:
                            # Add source file information to each page
                            for page in cache_data['pages']:
                                page['source_file'] = filename.replace('raw_results_', '').replace('.pkl', '')
                            all_pages_data.extend(cache_data['pages'])
                            logger.info(f"Successfully loaded cache file: {filename}")
                    except Exception as e:
                        logger.error(f"Error loading cache file {filename}: {str(e)}")
                        continue

            logger.info(f"Successfully loaded {len(all_pages_data)} pages from cache files")
            return all_pages_data
        except Exception as e:
            logger.error(f"Error reading cache files: {str(e)}")
            return [] 