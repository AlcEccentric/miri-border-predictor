#!/usr/bin/env python3
"""
R2 Data Downloader
Downloads data from Cloudflare R2 bucket to local test_data directory.
Equivalent to the Go r2_downloader.go script.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from r2_client import R2Client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class R2Downloader:
    def __init__(self, bucket_name: str, local_dir: str = "test_data"):
        """
        Initialize R2 downloader.
        
        Args:
            bucket_name: Name of the R2 bucket
            local_dir: Local directory to download files to
        """
        self.bucket_name = bucket_name
        self.local_dir = Path(local_dir)
        self.r2_client = R2Client()
        
        # Create local directory if it doesn't exist
        self.local_dir.mkdir(exist_ok=True)
    
    def download_file(self, key: str, local_path: Optional[Path] = None) -> bool:
        """
        Download a single file from R2 bucket.
        
        Args:
            key: R2 object key (path in bucket)
            local_path: Local file path. If None, uses key relative to local_dir
            
        Returns:
            True if successful, False otherwise
        """
        if local_path is None:
            local_path = self.local_dir / key
        
        # Create parent directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Downloading {key} to {local_path}")
            response = self.r2_client.get_object(self.bucket_name, key)
            
            with open(local_path, 'wb') as f:
                f.write(response['Body'].read())
            
            logger.info(f"Successfully downloaded {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            return False
    
    def download_prefix(self, prefix: str) -> int:
        """
        Download all files with a given prefix from R2 bucket.
        
        Args:
            prefix: Prefix to filter objects
            
        Returns:
            Number of files successfully downloaded
        """
        try:
            logger.info(f"Listing objects with prefix: {prefix}")
            response = self.r2_client.list_objects(self.bucket_name, prefix)
            
            if 'Contents' not in response:
                logger.warning(f"No objects found with prefix: {prefix}")
                return 0
            
            objects = response['Contents']
            logger.info(f"Found {len(objects)} objects to download")
            
            success_count = 0
            for obj in objects:
                key = obj['Key']
                if self.download_file(key):
                    success_count += 1
            
            logger.info(f"Successfully downloaded {success_count}/{len(objects)} files")
            return success_count
            
        except Exception as e:
            logger.error(f"Failed to list/download objects with prefix {prefix}: {e}")
            return 0
    
    def download_event_info(self) -> bool:
        """Download event info files."""
        logger.info("Downloading event info files...")
        return self.download_file("event_info/event_info_all.csv")
    
    def download_border_info(self, event_ids: Optional[list] = None) -> int:
        """
        Download border info files.
        
        Args:
            event_ids: List of event IDs to download. If None, downloads all.
            
        Returns:
            Number of files successfully downloaded
        """
        logger.info("Downloading border info files...")
        
        if event_ids:
            success_count = 0
            for event_id in event_ids:
                prefix = f"border_info/border_info_{event_id}_"
                success_count += self.download_prefix(prefix)
            return success_count
        else:
            return self.download_prefix("border_info/")
    
    def download_all(self, event_ids: Optional[list] = None) -> bool:
        """
        Download all necessary data files.
        
        Args:
            event_ids: List of event IDs to download border info for. If None, downloads all.
            
        Returns:
            True if all downloads successful, False otherwise
        """
        logger.info("Starting full data download...")
        
        # Download event info
        event_info_success = self.download_event_info()
        
        # Download border info
        border_info_count = self.download_border_info(event_ids)
        
        if event_info_success and border_info_count > 0:
            logger.info("All downloads completed successfully!")
            return True
        else:
            logger.error("Some downloads failed")
            return False


def main():
    """Main function to run the downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download data from R2 bucket')
    parser.add_argument('--bucket', required=True, help='R2 bucket name')
    parser.add_argument('--event-ids', nargs='*', type=int, help='Specific event IDs to download (optional)')
    parser.add_argument('--local-dir', default='test_data', help='Local directory to download to')
    parser.add_argument('--prefix', help='Download files with specific prefix only')
    
    args = parser.parse_args()
    
    downloader = R2Downloader(args.bucket, args.local_dir)
    
    if args.prefix:
        # Download specific prefix
        count = downloader.download_prefix(args.prefix)
        logger.info(f"Downloaded {count} files with prefix '{args.prefix}'")
    else:
        # Download all data
        success = downloader.download_all(args.event_ids)
        if not success:
            exit(1)


if __name__ == "__main__":
    main()