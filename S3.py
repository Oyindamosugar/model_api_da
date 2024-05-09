import boto3
import os

def list_all_files(bucket_name):
    """
    List all files in an S3 bucket.

    :param bucket_name: str - Name of the S3 bucket
    :return: list - A list of file keys in the bucket
    """
    # Create a session using environment variables for AWS credentials
    session = boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID','AKIA3FLD43Z6AZA2XKWT'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY', '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'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'eu-north-1')  # Default to us-east-1 if region not set
    )

    s3 = session.client('s3')

    # Initialize list to hold file keys
    file_keys = []

    # Paginate through results to handle large buckets
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        # Check if 'Contents' key is present in the response dictionary
        if 'Contents' in page:
            for obj in page['Contents']:
                file_keys.append(obj['Key'])

    return file_keys

# Usage Example
bucket = 'dissertationartefact'
# Replace with your S3 bucket name
files = list_all_files(bucket)
print("Files in S3 bucket:", files)