import logging
import os
import pickle
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
from io import BytesIO

logger = logging.getLogger(__name__)

class BlobStorageHelper:
    def __init__(self, connection_string: str, cache_dir: str = ".cache"):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.cache_dir = cache_dir
        self.cache_expiry = timedelta(days=1)
        

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_path(self, container_name: str) -> str:
        """Get the path to the cache file for a container."""
        return os.path.join(self.cache_dir, f"{container_name}_cache.pkl")
    
    def _load_cache(self, container_name: str) -> Optional[Dict]:
        """Load cached PDFs if available and not expired."""
        cache_path = self._get_cache_path(container_name)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if datetime.now() - cache_data['timestamp'] > self.cache_expiry:
                logger.info("Cache expired, will download fresh PDFs")
                return None
            
            logger.info(f"Loaded {len(cache_data['pdfs'])} PDFs from cache")
            return cache_data
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}")
            return None
    
    def _save_cache(self, container_name: str, pdf_files: List[Tuple[str, BytesIO]]):
        """Save PDFs to cache."""
        try:
            # Create a copy of BytesIO objects to avoid issues with position
            cached_pdfs = []
            for name, content in pdf_files:
                content_copy = BytesIO(content.getvalue())
                cached_pdfs.append((name, content_copy))
            
            cache_data = {
                'timestamp': datetime.now(),
                'pdfs': cached_pdfs
            }
            
            with open(self._get_cache_path(container_name), 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Saved {len(pdf_files)} PDFs to cache")
        except Exception as e:
            logger.warning(f"Error saving cache: {str(e)}")

    def download_pdfs_from_container(self, container_name: str) -> List[Tuple[str, BytesIO]]:
        """
        Downloads all PDF files from the specified container or loads from cache if available.
        Returns a list of tuples containing (filename, BytesIO object).
        """
        # Try to load from cache first
        cache_data = self._load_cache(container_name)
        if cache_data:
            # Reset file pointers to beginning
            for _, content in cache_data['pdfs']:
                content.seek(0)
            return cache_data['pdfs']
        
        # If no cache or expired, download from Azure
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            pdf_files = []

            blobs = container_client.list_blobs()
            for blob in blobs:
                if blob.name.lower().endswith('.pdf'):
                    logger.info(f"Downloading {blob.name}")
                    blob_client = container_client.get_blob_client(blob.name)
                    pdf_content = BytesIO()
                    pdf_content.write(blob_client.download_blob().readall())
                    pdf_content.seek(0)
                    
                    pdf_files.append((blob.name, pdf_content))

            logger.info(f"Downloaded {len(pdf_files)} PDF files")
            
            # Save to cache for future use
            self._save_cache(container_name, pdf_files)
            
            return pdf_files

        except Exception as e:
            logger.error(f"Error downloading PDFs: {str(e)}")
            raise