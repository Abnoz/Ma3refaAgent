import logging
from typing import List, Tuple
from azure.storage.blob import BlobServiceClient
from io import BytesIO

logger = logging.getLogger(__name__)

class BlobStorageHelper:
    def __init__(self, connection_string: str):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    def download_pdfs_from_container(self, container_name: str) -> List[Tuple[str, BytesIO]]:
        """
        Downloads all PDF files from the specified container.
        Returns a list of tuples containing (filename, BytesIO object).
        """
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
            return pdf_files

        except Exception as e:
            logger.error(f"Error downloading PDFs: {str(e)}")
            raise 