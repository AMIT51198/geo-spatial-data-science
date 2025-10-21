"""
Script to upload a local file to S3 bucket.
Set AWS credentials via environment variables or AWS CLI config.
"""

import boto3
import os

# Use environment variables for credentials
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('AWS_REGION', 'eu-north-1')

s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

local_file_path = 'path/to/local/file.txt'
bucket_name = 'india-schools-datalake'
s3_key = 'raw/india-schools/file.txt'  # S3 object key

try:
    s3.upload_file(local_file_path, bucket_name, s3_key)
    print("Upload complete!")
except Exception as e:
    print(f"Upload failed: {e}")
