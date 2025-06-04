import cv2
import os
import argparse
import shutil
parser = argparse.ArgumentParser(description='code for extracting frames from video')

parser.add_argument('--input_video', type=str, help='Video path to save result. See default for an e.g.', 
                                default='ai_core/wav2lip/output_preHD')

parser.add_argument('--frames_path', type=str, help='Video path to save result. See default for an e.g.', 
                                default='ai_core/wav2lip/frame_preHD')

args = parser.parse_args()
def extract_video_to_frame():
    shutil.rmtree(args.frames_path, ignore_errors=True)
    os.makedirs(args.frames_path, exist_ok=True)
    # Read the video file
    video_dir = args.input_video
    folder = os.listdir(video_dir)
    
    
    for file in folder:
        frame_index = 0
        video_path = f"{video_dir}/{file}"
        video = cv2.VideoCapture(video_path)
        os.makedirs(args.frames_path + f"/{file[:-4]}", exist_ok=True)
        # Get the frames per second (fps) and duration of the video
        fps = int(video.get(cv2.CAP_PROP_FPS))
        duration = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        
        # Loop through each frame of the video and save it as an image file
        for i in range(duration):
            ret, frame = video.read()
            if not ret:
                break
            frame_file = args.frames_path + f"/{file[:-4]}/frame_{frame_index:05d}.jpg"

            cv2.imwrite(frame_file, frame)
            frame_index += 1


        video.release()