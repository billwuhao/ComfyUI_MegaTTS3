# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
import pyloudnorm as pyln
from pydub import AudioSegment

# 设置FFmpeg路径
comfyui_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../../..'))
ffmpeg_path = os.path.join(comfyui_path, 'ffmpeg.exe')
ffprobe_path = os.path.join(comfyui_path, 'ffprobe.exe')

# 检查文件是否存在
if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
    # 设置pydub使用的FFmpeg路径
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path
else:
    print(f"警告: FFmpeg文件未找到! 尝试在以下路径查找:\n{ffmpeg_path}\n{ffprobe_path}")


def to_wav_bytes(wav, sr, norm=False):
    wav = wav.astype(float)
    if norm:
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -18.0)
        if np.abs(wav).max() >= 1:
            wav = wav / np.abs(wav).max() * 0.95
    wav = wav * 32767
    bytes_io = io.BytesIO()
    wavfile.write(bytes_io, sr, wav.astype(np.int16))
    return bytes_io.getvalue()


def save_wav(wav_bytes, path):
    with open(path[:-4] + '.wav', 'wb') as file:
        file.write(wav_bytes)
    if path[-4:] == '.mp3':
        to_mp3(path[:-4])


def to_mp3(out_path):
    if out_path[-4:] == '.wav':
        out_path = out_path[:-4]
    subprocess.check_call(
        f'ffmpeg -threads 1 -loglevel error -i "{out_path}.wav" -vn -b:a 192k -y -hide_banner -async 1 "{out_path}.mp3"',
        shell=True, stdin=subprocess.PIPE)
    subprocess.check_call(f'rm -f "{out_path}.wav"', shell=True)


def convert_to_wav(wav_path):
    # Check if the file exists
    if not os.path.exists(wav_path):
        print(f"The file '{wav_path}' does not exist.")
        return

    # Check if the file already has a .wav extension
    if not wav_path.endswith(".wav"):
        # Define the output path with a .wav extension
        out_path = os.path.splitext(wav_path)[0] + ".wav"

        # Load the audio file using pydub and convert it to WAV
        audio = AudioSegment.from_file(wav_path)
        audio.export(out_path, format="wav")

        print(f"Converted '{wav_path}' to '{out_path}'")


def convert_to_wav_bytes(audio_binary):
    # 直接返回原始字节流
    # 由于在这个模块中无法直接访问librosa，我们采用简单的方法
    byte_stream = io.BytesIO(audio_binary)
    # 重置光标位置到文件开头
    byte_stream.seek(0)
    return byte_stream


''' Smoothly combine audio segments using crossfade transitions." '''
def combine_audio_segments(segments, crossfade_duration=0.16, sr=24000):
    window_length = int(sr * crossfade_duration)
    hanning_window = np.hanning(2 * window_length)
    # Combine
    for i, segment in enumerate(segments):
        if i == 0:
            combined_audio = segment
        else:
            overlap = combined_audio[-window_length:] * hanning_window[window_length:] + segment[:window_length] * hanning_window[:window_length]
            combined_audio = np.concatenate(
                [combined_audio[:-window_length], overlap, segment[window_length:]]
            )
    return combined_audio