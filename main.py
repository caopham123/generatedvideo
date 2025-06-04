import os
import logging
logging.basicConfig(level=logging.WARNING)
from ai_core.REAL_ESRGAN.inference_realesrgan import frame_to_HD
from ai_core.tts.source.inference import OUTPUT_DIR, speak
from helpers.config_s3 import s3
from helpers.config_sqs import sqs
from ai_core.wav2lip.source.inference import add_lip
from ai_core.wav2lip.source.video2frames import extract_video_to_frame
import subprocess
import shutil
import time
from datetime import datetime
from helpers.common import get_env_var
import helpers.download_pretrained
import requests
import json
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageSequenceClip, AudioFileClip
import mimetypes
import gc

import cv2
import os

# Đặt biến môi trường để tắt logs DEBUG của botocore
os.makedirs("gfpgan", exist_ok=True)

def format_data():
    shutil.rmtree("ai_core/wav2lip/data/input/videos", ignore_errors=True)
    os.makedirs("ai_core/wav2lip/data/input/videos", exist_ok=True)
    shutil.rmtree("ai_core/wav2lip/data/input/frames", ignore_errors=True)
    os.makedirs("ai_core/wav2lip/data/input/frames", exist_ok=True)
    shutil.rmtree("ai_core/wav2lip/data/input/images", ignore_errors=True)
    os.makedirs("ai_core/wav2lip/data/input/images", exist_ok=True)
    shutil.rmtree("ai_core/REAL_ESRGAN/output", ignore_errors=True)
    os.makedirs("ai_core/REAL_ESRGAN/output", exist_ok=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shutil.rmtree("ai_core/wav2lip/data/output", ignore_errors=True)
    # shutil.rmtree("ai_core/tts/output/chunks", ignore_errors = True)
    os.makedirs("ai_core/tts/output/chunks", exist_ok=True)
    os.makedirs("ai_core/wav2lip/data/output/temp", exist_ok=True)
    shutil.rmtree("ai_core/wav2lip/output_preHD", ignore_errors=True)
    os.makedirs("ai_core/wav2lip/output_preHD", exist_ok=True)
    os.makedirs("ai_core/wav2lip/temp", exist_ok=True)
    

def concate_all_video(output_video_path):
    video_paths = [f"ai_core/wav2lip/data/output/temp/{file}" for file in os.listdir("ai_core/wav2lip/data/output/temp")]
    video_clips = [VideoFileClip(video_path) for video_path in video_paths]
    final_clip = concatenate_videoclips(video_clips, method="compose")
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

def convert_frames_to_video(input_folder="ai_core/REAL_ESRGAN/output", output_video_path="ai_core/wav2lip/data/output/output_video.mp4", fps = 50):
    for folder in os.listdir(input_folder):
        frame_folder = input_folder + "/" + folder
        audio_path = f"ai_core/tts/output/chunks/{folder}.wav"
        element_output_video_path = "ai_core/wav2lip/data/output/temp"
        ffmpeg_command = f'ffmpeg -r {fps} -i {frame_folder}/frame_%05d.jpg -i {audio_path} -c:v libx264 -crf 25 -preset veryslow -acodec aac -strict experimental {element_output_video_path}/{folder}.mp4'
        subprocess.run(ffmpeg_command, shell=True)
    
    gc.collect()
    concate_all_video(output_video_path)


def text_to_video(text, character_url, data, speed = 1, vocal = "female", language = "VI", fps = 50):
    if text[-1] != ".":
        text = text + "."
    format_data()
    
    data = list(data)
    if len(data) != 0:
        type_input = "video"
        index = 0
        for obj in data:
            url = obj["url"]
            weight = int(obj["weight"])
            for i in range(weight):
                file_local =  "ai_core/wav2lip/data/input/videos/video_" + str(index) + ".mp4"
                response = requests.get(url)
                if response.status_code == 200:
                    open(file_local, "wb").write(response.content)
                    
                cap = cv2.VideoCapture(file_local)
                index_frame = 0
                while 1:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    os.makedirs("ai_core/wav2lip/data/input/frames/video_" + str(index), exist_ok=True)
                    frame_path = "ai_core/wav2lip/data/input/frames/video_" + str(index) + '/' + f'frame_{index_frame:05d}.jpg'
                    cv2.imwrite(frame_path, frame)
                    index_frame += 1
                index += 1 
                cap.release()

    else:
        type_input = "image"
        file_local = "ai_core/wav2lip/data/input/images/image.jpg"
        response = requests.get(character_url)
        if response.status_code == 200:
            open(file_local, "wb").write(response.content)

    speak(text, speed = speed, vocal = vocal, language = language)
    gc.collect()
    add_lip(type_input, custom_fps = fps) 
    extract_video_to_frame()
    frame_to_HD()
    convert_frames_to_video(fps = fps)
    return "ai_core/wav2lip/data/output/output_video.mp4"




def main():
    while True:
        response = sqs.receive_message(
            QueueUrl=get_env_var("s3", "AWS_QUEUE_INPUT_URL"),
            MaxNumberOfMessages=1,
            VisibilityTimeout=0,
            WaitTimeSeconds=0
        )
        if 'Messages' in response:
            received_message = response['Messages'][0]
            receipt_handle = received_message['ReceiptHandle']
            parsed_body = json.loads(received_message["Body"])
            print(parsed_body)
            video_id = parsed_body["video_id"]
            text_input = parsed_body["text_input"]
            image_url = parsed_body["character_url"]
            data = parsed_body["data"]
            speed = float(parsed_body["speed"])
            language = parsed_body["language"]
            vocal = parsed_body["vocal"]
            sqs.delete_message(QueueUrl=get_env_var("s3", "AWS_QUEUE_INPUT_URL"), ReceiptHandle=receipt_handle)
            try:
                video_output_file_name = text_to_video(text_input, image_url,data, speed, vocal, language)
                # Get the current date and time
                current_datetime = datetime.now()
                # Extract the date component
                current_date = current_datetime.date()
                video_output_file_name_s3 = f"video/{current_date}/{video_id}.mp4"
                metadata = {'format': 'mp4'}
                content_type, _ = mimetypes.guess_type(video_output_file_name_s3)
                metadata['Content-Type'] = content_type
                with open(video_output_file_name, "rb") as f:
                    s3.upload_fileobj(f, get_env_var('s3', 'AWS_S3_BUCKET_NAME'), video_output_file_name_s3, ExtraArgs={'Metadata': metadata})
                
                message_body = {
                    'video_id': video_id,
                    'video_url': f"https://{get_env_var('s3', 'AWS_S3_BUCKET_NAME')}.s3.{get_env_var('s3', 'AWS_REGION_NAME')}.amazonaws.com/{video_output_file_name_s3}",
                    "status": "done",
                    "content_type": "mp4"

                }
                # Xác nhận xử lý xong thông điệp để nó không xuất hiện trong hàng đợi nữa
                
                request = sqs.send_message(
                    QueueUrl=get_env_var("s3", "AWS_QUEUE_OUTPUT_URL"),
                    MessageBody=json.dumps(message_body)  # Convert the dict to a JSON string before sending
                )
                
            except Exception as e:
                message_body = {
                    'video_id': video_id,
                    'video_url': None,
                    "status": f"failed"
                }
                os.makedirs("logs", exist_ok= True)
                with open(f"logs/{video_id}.txt", "w", encoding="utf-8") as f:
                    f.write(str(e))
                request = sqs.send_message(
                    QueueUrl=get_env_var("s3", "AWS_QUEUE_OUTPUT_URL"),
                    MessageBody=json.dumps(message_body)  # Convert the dict to a JSON string before sending
                )
            time.sleep(5)
        else:
            print("No messages in the queue. Waiting for 30 seconds...")
            time.sleep(30)  # Chờ 30 giây trước khi thử lại

print("Docker build done!")
# data = """The critical shortage of IT engineers especially in Japan motivated us to start Tokyo Techies in Japan and Tokyo Tech Lab in Vietnam with the mission of bridging the gap between high demand and low supply of tech talents. 
# We help enterprises to stay competitive in the global market by providing them with One-stop IT Consulting and Development Support to realize their ideas into world class products and services, then help them to build in-house engineering capacity for the long-term.
# With a team of experienced Techies from all around the world, we want to make everyone’s lives easier by constantly trying to achieve automation and optimization in everything we build."""
main()
# convert_frames_to_video()
# speak("Tao là bố mày là con tao", vocal="male_north_side")
# text_to_video(text=data, character_url = "", data = [], speed = 1, vocal = "male", language = "EN", fps = 50)
# convert_frames_to_video()