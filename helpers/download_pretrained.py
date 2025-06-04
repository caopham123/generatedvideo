from .common import get_env_var
from .config_s3 import s3
import os
import requests

s3_prefix = 'ai_model'

os.makedirs("ai_core/REAL_ESRGAN/weights", exist_ok= True)
os.makedirs("ai_core/tts/model/VI/female", exist_ok= True)
os.makedirs("ai_core/tts/model/VI/male", exist_ok= True)
os.makedirs("ai_core/wav2lip/model/Wav2LIP", exist_ok= True)
os.makedirs("ai_core/wav2lip/model/experiments/001_ESRGAN_x4_f64b23_custom16k_500k_B16G1_wandb/models", exist_ok= True)
os.makedirs("gfpgan/weights", exist_ok= True)

response = s3.list_objects_v2(Bucket=get_env_var("s3", "AWS_S3_BUCKET_NAME"), Prefix=s3_prefix)

# Extract file names from the response
file_names = [obj['Key'] for obj in response.get('Contents', [])]

# Print the list of file names

for file_name in file_names:
    url_model = f"https://{get_env_var('s3', 'AWS_S3_BUCKET_NAME')}.s3.{get_env_var('s3', 'AWS_REGION_NAME')}.amazonaws.com/{file_name}"
    file_local = file_name.replace("ai_model", "ai_core")
    if os.path.exists(file_local) is False:
        print(f"Downloading: {url_model} to {file_local} ")
        response = requests.get(url_model)
        if response.status_code == 200:
            open(file_local, "wb").write(response.content)

