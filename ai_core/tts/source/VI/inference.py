import time
import torch  # isort:skip
import shutil
torch.manual_seed(42)
import json
import re
import unicodedata
from types import SimpleNamespace
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import regex
import os
from .models import DurationNet, SynthesizerTrn
import pydub

device = "cuda" if torch.cuda.is_available() else "cpu"
hps = None
phone_set = None
sil_idx = None
# load phone set json file


OUTPUT_DIR = "ai_core/tts/output/chunks"

space_re = regex.compile(r"\s+")
number_re = regex.compile("([0-9]+)")
digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
num_re = regex.compile(r"([0-9.,]*[0-9])")
alphabet = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵbcdđghklmnpqrstvx"
keep_text_and_num_re = regex.compile(rf"[^\s{alphabet}.,0-9]")
keep_text_re = regex.compile(rf"[^\s{alphabet}]")


def read_number(num: str) -> str:
    if len(num) == 1:
        return digits[int(num)]
    elif len(num) == 2 and num.isdigit():
        n = int(num)
        end = digits[n % 10]
        if n == 10:
            return "mười"
        if n % 10 == 5:
            end = "lăm"
        if n % 10 == 0:
            return digits[n // 10] + " mươi"
        elif n < 20:
            return "mười " + end
        else:
            if n % 10 == 1:
                end = "mốt"
            return digits[n // 10] + " mươi " + end
    elif len(num) == 3 and num.isdigit():
        n = int(num)
        if n % 100 == 0:
            return digits[n // 100] + " trăm"
        elif num[1] == "0":
            return digits[n // 100] + " trăm lẻ " + digits[n % 100]
        else:
            return digits[n // 100] + " trăm " + read_number(num[1:])
    elif len(num) >= 4 and len(num) <= 6 and num.isdigit():
        n = int(num)
        n1 = n // 1000
        return read_number(str(n1)) + " ngàn " + read_number(num[-3:])
    elif "," in num:
        n1, n2 = num.split(",")
        return read_number(n1) + " phẩy " + read_number(n2)
    elif "." in num:
        parts = num.split(".")
        if len(parts) == 2:
            if parts[1] == "000":
                return read_number(parts[0]) + " ngàn"
            elif parts[1].startswith("00"):
                end = digits[int(parts[1][2:])]
                return read_number(parts[0]) + " ngàn lẻ " + end
            else:
                return read_number(parts[0]) + " ngàn " + read_number(parts[1])
        elif len(parts) == 3:
            return (
                read_number(parts[0])
                + " triệu "
                + read_number(parts[1])
                + " ngàn "
                + read_number(parts[2])
            )
    return num


def text_to_phone_idx(text):
    # lowercase
    text = text.lower()
    # unicode normalize
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace(";", " ; ")
    text = text.replace(":", " : ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace("(", " ( ")

    text = num_re.sub(r" \1 ", text)
    words = text.split()
    words = [read_number(w) if num_re.fullmatch(w) else w for w in words]
    text = " ".join(words)

    # remove redundant spaces
    text = re.sub(r"\s+", " ", text)
    # remove leading and trailing spaces
    text = text.strip()
    # convert words to phone indices
    tokens = []
    for c in text:
        # if c is "," or ".", add <sil> phone
        if c in ":,.!?;(":
            tokens.append(sil_idx)
        elif c in phone_set:
            tokens.append(phone_set.index(c))
        elif c == " ":
            # add <sep> phone
            tokens.append(0)
    if tokens[0] != sil_idx:
        # insert <sil> phone at the beginning
        tokens = [sil_idx, 0] + tokens
    if tokens[-1] != sil_idx:
        tokens = tokens + [0, sil_idx]
    return tokens


def text_to_speech(duration_net, generator, text):
    # prevent too long text
    if len(text) > 500:
        text = text[:500]

    phone_idx = text_to_phone_idx(text)
    batch = {
        "phone_idx": np.array([phone_idx]),
        "phone_length": np.array([len(phone_idx)]),
    }

    # predict phoneme duration
    phone_length = torch.from_numpy(batch["phone_length"].copy()).long().to(device)
    phone_idx = torch.from_numpy(batch["phone_idx"].copy()).long().to(device)
    with torch.inference_mode():
        phone_duration = duration_net(phone_idx, phone_length)[:, :, 0] * 1000
    phone_duration = torch.where(
        phone_idx == sil_idx, torch.clamp_min(phone_duration, 200), phone_duration
    )
    phone_duration = torch.where(phone_idx == 0, 0, phone_duration)

    # generate waveform
    end_time = torch.cumsum(phone_duration, dim=-1)
    start_time = end_time - phone_duration
    start_frame = start_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
    end_frame = end_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
    spec_length = end_frame.max(dim=-1).values
    pos = torch.arange(0, spec_length.item(), device=device)
    attn = torch.logical_and(
        pos[None, :, None] >= start_frame[:, None, :],
        pos[None, :, None] < end_frame[:, None, :],
    ).float()
    with torch.inference_mode():
        y_hat = generator.infer(
            phone_idx, phone_length, spec_length, attn, max_len=None, noise_scale=0.667
        )[0]
    wave = y_hat[0, 0].data.cpu().numpy()
    return (wave * (2**15)).astype(np.int16)


def load_models(language, vocal):
    global hps, phone_set, sil_idx
    config_file = f"ai_core/tts/model/{language}/{vocal}/config.json"
    duration_model_path = f"ai_core/tts/model/{language}/{vocal}/duration_model.pth"
    lightspeed_model_path = f"ai_core/tts/model/{language}/{vocal}/gen_model.pth"
    phone_set_file = f"ai_core/tts/model/{language}/{vocal}/phone_set.json"
    
    with open(phone_set_file, "r") as f:
        phone_set = json.load(f)
    with open(config_file, "rb") as f:
        hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
    
    assert phone_set[0][1:-1] == "SEP"
    assert "sil" in phone_set
    sil_idx = phone_set.index("sil")
    
    
    duration_net = DurationNet(hps.data.vocab_size, 64, 4).to(device)
    duration_net.load_state_dict(torch.load(duration_model_path, map_location=device))
    duration_net = duration_net.eval()
    generator = SynthesizerTrn(
        hps.data.vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **vars(hps.model),
    ).to(device)
    del generator.enc_q
    ckpt = torch.load(lightspeed_model_path, map_location=device)
    params = {}
    for k, v in ckpt["net_g"].items():
        k = k[7:] if k.startswith("module.") else k
        params[k] = v
    generator.load_state_dict(params, strict=False)
    del ckpt, params
    generator = generator.eval()
    return duration_net, generator


def speak_VI(text, speed = 1, language = "VI", vocal = "female", ):
    
    duration_net, generator = load_models(language, vocal)
    paragraphs = text.split(".")
    for paragraph in paragraphs:
        output_file = OUTPUT_DIR + '/' + str(time.time()) + ".wav"
        clips = []  # list of audio clips
    # silence = np.zeros(hps.data.sampling_rate // 4)
        paragraph = paragraph.strip()
        if paragraph == "":
            continue
        clips.append(text_to_speech(duration_net, generator, paragraph))
        # clips.append(silence)
        y = np.concatenate(clips)
        
        sf.write(output_file, y, int(hps.data.sampling_rate * speed))
    
    folder = os.listdir("ai_core/tts/output/chunks")
    file_names = ["ai_core/tts/output/chunks/" + file_name for file_name in folder]
    audio_segments = [AudioSegment.from_file(file_name) for file_name in file_names]
    
    audio = sum(audio_segments)
    audio.export("ai_core/wav2lip/data/output/output_audio.wav", format="wav")
    return "ai_core/wav2lip/data/output/output_audio.wav"





