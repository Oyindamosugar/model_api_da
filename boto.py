import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

bucket_name = 'model'
region_name = 'eu-north-1'

try:
    # Create an S3 client specifying the region
    s3_client = boto3.client('s3', region_name=region_name)
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page['Contents']:
            print(obj['Key'])
except NoCredentialsError:
    print("Credentials not available")
except PartialCredentialsError:
    print("Incomplete credentials")
except ClientError as e:
    if e.response['Error']['Code'] == 'AccessDenied':
        print("Access denied. Check the IAM permissions.")
    else:
        print("Unexpected error: %s" % e)
