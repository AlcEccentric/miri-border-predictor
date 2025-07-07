import os
import boto3
from typing import Optional
from dotenv import load_dotenv


class R2Client:
    """A client for interacting with Cloudflare R2 storage."""
    
    def __init__(self, 
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 endpoint_url: Optional[str] = None):
        """
        Initialize R2Client with credentials.
        
        Args:
            aws_access_key_id: R2 access key ID. If None, will read from env vars or .env file
            aws_secret_access_key: R2 secret access key. If None, will read from env vars or .env file  
            endpoint_url: R2 endpoint URL. If None, will read from env vars or .env file
        """
        # Try to load from .env file if it exists (for local development)
        if os.path.exists('.env'):
            load_dotenv()
        
        # Use provided credentials or fall back to environment variables
        self.aws_access_key_id = aws_access_key_id or os.getenv('R2_ACCESS_KEY_ID')
        self.aws_secret_access_key = aws_secret_access_key or os.getenv('R2_SECRET_ACCESS_KEY')
        self.endpoint_url = endpoint_url or os.getenv('R2_ENDPOINT_URL')
        
        if not all([self.aws_access_key_id, self.aws_secret_access_key, self.endpoint_url]):
            raise ValueError(
                "Missing R2 credentials. Please provide them as parameters or set "
                "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_ENDPOINT_URL environment variables."
            )
        
        self.client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.endpoint_url
        )
    
    def get_object(self, bucket_name: str, key: str):
        """Get an object from R2 storage."""
        return self.client.get_object(Bucket=bucket_name, Key=key)
    
    def list_objects(self, bucket_name: str, prefix: str = ''):
        """List objects in R2 storage with optional prefix."""
        return self.client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    def put_object(self, bucket_name: str, key: str, body, **kwargs):
        """Put an object to R2 storage."""
        return self.client.put_object(Bucket=bucket_name, Key=key, Body=body, **kwargs)
    
    def delete_object(self, bucket_name: str, key: str):
        """Delete an object from R2 storage."""
        return self.client.delete_object(Bucket=bucket_name, Key=key)
