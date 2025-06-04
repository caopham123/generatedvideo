### Install amd64 emulator
# docker run --privileged --rm tonistiigi/binfmt --install amd64 
### Build docker image
# docker buildx build --platform=linux/amd64 -t tims-ai .
### Run docker container
# docker run --platform=linux/amd64 -p 9000:9000 --name=tims-ai tims-ai
#model arcFace for Face recognition: https://drive.google.com/file/d/1cLIh-n_q_R7yJ-rpLM4n2IbVltxLm5YN/view?usp=sharing
# pull docker image
FROM --platform=linux/amd64 python:3.10 AS COMPILE-IMAGE
# set work directory
WORKDIR /app
# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# install dependencies
RUN pip install --user --upgrade pip && pip3 install --no-compile --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
COPY ./requirements.txt ./requirements.txt
COPY ./ai_core/wav2lip/requirements.txt ./c
COPY ./ai_core/REAL_ESRGAN/requirements.txt ./ai_core/REAL_ESRGAN/requirements.txt
RUN pip install --no-compile --user -r ./requirements.txt
RUN pip install --no-compile --user -r ./ai_core/wav2lip/requirements.txt
RUN pip install --no-compile --user -r ./ai_core/REAL_ESRGAN/requirements.txt

FROM --platform=linux/amd64 python:3.10 AS BUILD-IMAGE
RUN apt-get update
RUN apt-cache search espeak
RUN apt-get install -y espeak
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 ffmpeg 

COPY --from=COMPILE-IMAGE /root/.local /root/.local

# set work directory
WORKDIR /app
ENV PATH="/root/.local/bin:$PATH"
# copy project
COPY . .
CMD python3 main.py