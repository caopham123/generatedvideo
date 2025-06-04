import time


import os
import json
import math
import torch

from pydub import AudioSegment
from . import commons
from . import utils
import soundfile as sf
from .models import SynthesizerTrn
from .text.symbols import symbols
from .text import text_to_sequence


OUTPUT_DIR = "ai_core/tts/output/chunks"
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def speak_EN(text:str, speed: float = 1.0, vocal:str = "male"):
    
    hps = utils.get_hparams_from_file(f"ai_core/tts/model/EN/{vocal}/config.json")
    
    if vocal == "female":
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).cuda()
        _ = net_g.eval()

        _ = utils.load_checkpoint("ai_core/tts/model/EN/female/gen_model.pth", net_g, None)   
        
    else:
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model).cuda()
        _ = net_g.eval()
        _ = utils.load_checkpoint("ai_core/tts/model/EN/male/gen_model.pth", net_g, None)
        
    paragraphs = text.split(".")
    for paragraph in paragraphs[:-1]:
        output_file = OUTPUT_DIR + '/' + str(time.time()) + ".wav"
        stn_tst = get_text(paragraph, hps)
        with torch.no_grad():
            if vocal == "female": 
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
                audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            else: 
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
                sid = torch.LongTensor([4]).cuda()
                audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            sf.write(output_file, audio, int(hps.data.sampling_rate * speed))
        
    folder = os.listdir("ai_core/tts/output/chunks")
    file_names = ["ai_core/tts/output/chunks/" + file_name for file_name in folder]
    audio_segments = [AudioSegment.from_file(file_name) for file_name in file_names]
    
    audio = sum(audio_segments)
    audio.export("ai_core/wav2lip/data/output/output_audio.wav", format="wav")
    del net_g, hps, audio, audio_segments
    return "ai_core/wav2lip/data/output/output_audio.wav"
