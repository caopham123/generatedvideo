import boto3
from .common import get_env_var

queue_url = 'https://sqs.ap-southeast-1.amazonaws.com/622742764907/myqueue'

sqs = boto3.client('sqs', region_name=get_env_var("s3", "AWS_REGION_NAME"), aws_access_key_id=get_env_var("s3", "AWS_ACCESS_KEY_ID"), aws_secret_access_key=get_env_var("s3", "AWS_SECRET_ACCESS_KEY"))

