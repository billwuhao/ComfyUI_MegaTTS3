import json
import os
import librosa
import numpy as np
import torch
import torchaudio
from typing import List, Union, Optional, Dict
import re
import logging

# from tn.chinese.normalizer import Normalizer as ZhNormalizer
# from tn.english.normalizer import Normalizer as EnNormalizer
# from langdetect import detect as classify_language
import pyloudnorm as pyln
import folder_paths

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator
from tts.frontend_function import g2p, align, make_dur_prompt, dur_pred, prepare_inputs_for_dit
from tts.utils.audio_utils.io import save_wav, to_wav_bytes, convert_to_wav_bytes, combine_audio_segments
from tts.utils.commons.ckpt_utils import load_ckpt
from tts.utils.commons.hparams import set_hparams, hparams
from tts.utils.text_utils.text_encoder import TokenTextEncoder
from tts.utils.text_utils.split_text import chunk_text_chinese, chunk_text_english
from tts.utils.commons.hparams import hparams, set_hparams


models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")

# if "TOKENIZERS_PARALLELISM" not in os.environ:
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"

# def convert_to_wav(wav_path):
#     # Check if the file exists
#     if not os.path.exists(wav_path):
#         print(f"The file '{wav_path}' does not exist.")
#         return

#     # Check if the file already has a .wav extension
#     if not wav_path.endswith(".wav"):
#         # Define the output path with a .wav extension
#         out_path = os.path.splitext(wav_path)[0] + ".wav"

#         # Load the audio file using pydub and convert it to WAV
#         audio = AudioSegment.from_file(wav_path)
#         audio.export(out_path, format="wav")

#         print(f"Converted '{wav_path}' to '{out_path}'")


# def cut_wav(wav_path, max_len=28):
#     audio = AudioSegment.from_file(wav_path)
#     audio = audio[:int(max_len * 1000)]
#     audio.export(wav_path, format="wav")


# def audio_tensor_to_wavbytes(audio_tensor, sample_rate=24000):
#     if len(audio_tensor.shape) > 1:
#         audio_tensor = audio_tensor.mean(dim=0)  # 取平均转为单通道
#     # 转换为 numpy 数组并调整范围
#     audio_data = audio_tensor.cpu().numpy()
#     if audio_data.max() <= 1.0:
#         audio_data = (audio_data * 32767).astype(np.int16)
    
#     # 创建 WAV 缓冲区
#     wav_buffer = io.BytesIO()
#     wavfile.write(wav_buffer, sample_rate, audio_data)

#     return wav_buffer.getvalue()

def get_all_files(
    root_dir: str,
    return_type: str = "list",
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    relative_path: bool = False
) -> Union[List[str], dict]:
    """
    递归获取目录下所有文件路径
    
    :param root_dir: 要遍历的根目录
    :param return_type: 返回类型 - "list"(列表) 或 "dict"(按目录分组)
    :param extensions: 可选的文件扩展名过滤列表 (如 ['.py', '.txt'])
    :param exclude_dirs: 要排除的目录名列表 (如 ['__pycache__', '.git'])
    :param relative_path: 是否返回相对路径 (相对于root_dir)
    :return: 文件路径列表或字典
    """
    file_paths = []
    file_dict = {}
    
    # 规范化目录路径
    root_dir = os.path.normpath(root_dir)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 处理排除目录
        if exclude_dirs:
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        current_files = []
        for filename in filenames:
            # 扩展名过滤
            if extensions:
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue
            
            # 构建完整路径
            full_path = os.path.join(dirpath, filename)
            
            # 处理相对路径
            if relative_path:
                full_path = os.path.relpath(full_path, root_dir)
            
            current_files.append(full_path)
        
        if return_type == "dict":
            # 使用相对路径或绝对路径作为键
            dict_key = os.path.relpath(dirpath, root_dir) if relative_path else dirpath
            if current_files:
                file_dict[dict_key] = current_files
        else:
            file_paths.extend(current_files)
    
    return file_dict if return_type == "dict" else file_paths

def get_speakers():
    speakers_dir = os.path.join(model_path, "MegaTTS3", "speakers")
    if not os.path.exists(speakers_dir):
        os.makedirs(speakers_dir, exist_ok=True)
        return []
    speakers = get_all_files(speakers_dir, extensions=[".wav"], relative_path=True)
    return speakers


class SimpleNormalizer:
    # Multi-character sequences (Applied FIRST)
    _MULTI_CHAR_REP_MAP: Dict[str, str] = {
        "...": "...", # Normalize to standard ellipsis first
        "......": "...",
        "…": "...", # Map unicode ellipsis to dots
    }

    # --- Language-Specific Maps (used in logic) ---

    # Chinese digit mapping
    _ZH_DIGIT_MAP: Dict[str, str] = {
        "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
        "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
    }


    def __init__(self, lang: str = "zh"):
        """
        Initializes the SimpleNormalizer.

        Args:
            lang (str): The target language ('zh' or 'en'). Defaults to 'zh'.
        """
        if lang not in ["zh", "en"]:
            raise ValueError(f"Unsupported language: {lang}. Supported: 'zh', 'en'")
        self.lang = lang

    def _zh_digit_replacer(self, match: re.Match) -> str:
        """Helper function to replace a matched digit with its Chinese char."""
        return self._ZH_DIGIT_MAP[match.group(0)]

    def normalize(self, text: str) -> str:
        """
        Performs text normalization based on the initialized language.

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.
        """
        if not isinstance(text, str):
            logging.warning(f"Input is not a string: {type(text)}. Returning as is.")
            return text
        if not text:
            return ""

        # --- Normalization Steps ---

        # Basic cleanup (Null bytes)
        text = text.replace('\x00', '')

        # Multi-character replacements (Ellipsis normalization - applied first)
        for old, new in self._MULTI_CHAR_REP_MAP.items():
            text = text.replace(old, new)

        #  Language-specific processing
        if self.lang == "zh":
            text = re.sub(r'(?<=\d)\.(?=\d)', '点', text) # e.g., "1.2" -> "1点2"
            text = re.sub(r'\d', self._zh_digit_replacer, text) # e.g., "1点2" -> "一点二"

        elif self.lang == "en":
            text = text.lower()
            # Note: '.' remains '.' because it wasn't converted in step 3 or ZH steps.

        text = re.sub(r'\s+', ' ', text).strip()

        return text


class MegaTTS3DiTInfer():
    def __init__(
            self, 
            device=None,
            ckpt_root=os.path.join(model_path, "MegaTTS3"),
            dit_exp_name='diffusion_transformer',
            frontend_exp_name='aligner_lm',
            wavvae_exp_name='wavvae',
            dur_ckpt_path='duration_lm',
            g2p_exp_name='g2p',
            precision=torch.float16,
            **kwargs
        ):
        self.sr = 24000
        self.fm = 8
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.precision = precision

        # build models
        self.dit_exp_name = os.path.join(ckpt_root, dit_exp_name)
        self.frontend_exp_name = os.path.join(ckpt_root, frontend_exp_name)
        self.wavvae_exp_name = os.path.join(ckpt_root, wavvae_exp_name)
        self.dur_exp_name = os.path.join(ckpt_root, dur_ckpt_path)
        self.g2p_exp_name = os.path.join(ckpt_root, g2p_exp_name)
        self.build_model(self.device)

        # init text normalizer
        # self.zh_normalizer = ZhNormalizer(overwrite_cache=False, remove_erhua=False, remove_interjections=False)
        # self.en_normalizer = EnNormalizer(overwrite_cache=False)

        self.zh_normalizer = SimpleNormalizer(lang='zh')
        self.en_normalizer = SimpleNormalizer(lang='en')

        # loudness meter
        self.loudness_meter = pyln.Meter(self.sr)
        
    def clean(self):
        import gc
        self.dur_model = None
        self.dit= None
        self.g2p_model = None
        self.wavvae = None
        gc.collect()
        torch.cuda.empty_cache()

    def build_model(self, device):
        set_hparams(exp_name=self.dit_exp_name, print_hparams=False)

        ''' Load Dict '''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ling_dict = json.load(open(f"{current_dir}/tts/utils/text_utils/dict.json", encoding='utf-8-sig'))
        self.ling_dict = {k: TokenTextEncoder(None, vocab_list=ling_dict[k], replace_oov='<UNK>') for k in ['phone', 'tone']}
        self.token_encoder = token_encoder = self.ling_dict['phone']
        ph_dict_size = len(token_encoder)

        ''' Load Duration LM '''
        from tts.modules.ar_dur.ar_dur_predictor import ARDurPredictor
        hp_dur_model = self.hp_dur_model = set_hparams(f'{self.dur_exp_name}/config.yaml', global_hparams=False)
        hp_dur_model['frames_multiple'] = hparams['frames_multiple']
        self.dur_model = ARDurPredictor(
            hp_dur_model, hp_dur_model['dur_txt_hs'], hp_dur_model['dur_model_hidden_size'],
            hp_dur_model['dur_model_layers'], ph_dict_size,
            hp_dur_model['dur_code_size'],
            use_rot_embed=hp_dur_model.get('use_rot_embed', False))
        self.length_regulator = LengthRegulator()
        load_ckpt(self.dur_model, f'{self.dur_exp_name}', 'dur_model')
        self.dur_model.eval()
        self.dur_model.to(device)

        ''' Load Diffusion Transformer '''
        from tts.modules.llm_dit.dit import Diffusion
        self.dit = Diffusion()
        load_ckpt(self.dit, f'{self.dit_exp_name}', 'dit', strict=False)
        self.dit.eval()
        self.dit.to(device)
        self.cfg_mask_token_phone = 302 - 1
        self.cfg_mask_token_tone = 32 - 1

        ''' Load Frontend LM '''
        from tts.modules.aligner.whisper_small import Whisper
        self.aligner_lm = Whisper()
        load_ckpt(self.aligner_lm, f'{self.frontend_exp_name}', 'model')
        self.aligner_lm.eval()
        self.aligner_lm.to(device)
        self.kv_cache = None
        self.hooks = None

        ''' Load G2P LM'''
        from transformers import AutoTokenizer, AutoModelForCausalLM
        g2p_tokenizer = AutoTokenizer.from_pretrained(self.g2p_exp_name, padding_side="right")
        g2p_tokenizer.padding_side = "right"
        self.g2p_model = AutoModelForCausalLM.from_pretrained(self.g2p_exp_name).eval().to(device)
        self.g2p_tokenizer = g2p_tokenizer
        self.speech_start_idx = g2p_tokenizer.encode('<Reserved_TTS_0>')[0]

        ''' Wav VAE '''
        self.hp_wavvae = hp_wavvae = set_hparams(f'{self.wavvae_exp_name}/config.yaml', global_hparams=False)
        from tts.modules.wavvae.decoder.wavvae_v3 import WavVAE_V3
        self.wavvae = WavVAE_V3(hparams=hp_wavvae)
        if os.path.exists(f'{self.wavvae_exp_name}/model_only_last.ckpt'):
            load_ckpt(self.wavvae, f'{self.wavvae_exp_name}/model_only_last.ckpt', 'model_gen', strict=True)
            self.has_vae_encoder = True
        else:
            load_ckpt(self.wavvae, f'{self.wavvae_exp_name}/decoder.ckpt', 'model_gen', strict=False)
            self.has_vae_encoder = False
        self.wavvae.eval()
        self.wavvae.to(device)
        self.vae_stride = hp_wavvae.get('vae_stride', 4)
        self.hop_size = hp_wavvae.get('hop_size', 4)
    
    def preprocess(self, audio_bytes, latent_file=None, topk_dur=1, **kwargs):
        wav_bytes = convert_to_wav_bytes(audio_bytes)

        ''' Load wav '''
        wav, _ = librosa.core.load(wav_bytes, sr=self.sr)
        # Pad wav if necessary
        ws = hparams['win_size']
        if len(wav) % ws < ws - 1:
            wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
        wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
        self.loudness_prompt = self.loudness_meter.integrated_loudness(wav.astype(float))

        ''' obtain alignments with aligner_lm '''
        ph_ref, tone_ref, mel2ph_ref = align(self, wav)

        with torch.inference_mode():
            ''' Forward WaveVAE to obtain: prompt latent '''
            if self.has_vae_encoder:
                wav = torch.FloatTensor(wav)[None].to(self.device)
                vae_latent = self.wavvae.encode_latent(wav)
                vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
            else:
                assert latent_file is not None, "Please provide latent_file in WaveVAE decoder-only mode"
                vae_latent = torch.from_numpy(np.load(latent_file)).to(self.device)
                vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
        
            ''' Duration Prompting '''
            self.dur_model.hparams["infer_top_k"] = topk_dur if topk_dur > 1 else None
            incremental_state_dur_prompt, ctx_dur_tokens = make_dur_prompt(self, mel2ph_ref, ph_ref, tone_ref)
            
        return {
            'ph_ref': ph_ref,
            'tone_ref': tone_ref,
            'mel2ph_ref': mel2ph_ref,
            'vae_latent': vae_latent,
            'incremental_state_dur_prompt': incremental_state_dur_prompt,
            'ctx_dur_tokens': ctx_dur_tokens,
        }

    def forward(self, resource_context, input_text, language_type, time_step, p_w, t_w, dur_disturb=0.1, dur_alpha=1.0, **kwargs):
        device = self.device

        ph_ref = resource_context['ph_ref'].to(device)
        tone_ref = resource_context['tone_ref'].to(device)
        mel2ph_ref = resource_context['mel2ph_ref'].to(device)
        vae_latent = resource_context['vae_latent'].to(device)
        ctx_dur_tokens = resource_context['ctx_dur_tokens'].to(device)
        incremental_state_dur_prompt = resource_context['incremental_state_dur_prompt']

        with torch.inference_mode():
            ''' Generating '''
            wav_pred_ = []
            # language_type = classify_language(input_text)
            if language_type == 'en':
                input_text = self.en_normalizer.normalize(input_text)
                text_segs = chunk_text_english(input_text, max_chars=130)
            else:
                input_text = self.zh_normalizer.normalize(input_text)
                text_segs = chunk_text_chinese(input_text, limit=60)

            for seg_i, text in enumerate(text_segs):
                ''' G2P '''
                ph_pred, tone_pred = g2p(self, text)

                ''' Duration Prediction '''
                mel2ph_pred = dur_pred(self, ctx_dur_tokens, incremental_state_dur_prompt, ph_pred, tone_pred, seg_i, dur_disturb, dur_alpha, is_first=seg_i==0, is_final=seg_i==len(text_segs)-1)
                
                inputs = prepare_inputs_for_dit(self, mel2ph_ref, mel2ph_pred, ph_ref, tone_ref, ph_pred, tone_pred, vae_latent)
                # Speech dit inference
                with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                    x = self.dit.inference(inputs, timesteps=time_step, seq_cfg_w=[p_w, t_w]).float()
                
                # WavVAE decode
                x[:, :vae_latent.size(1)] = vae_latent
                wav_pred = self.wavvae.decode(x)[0,0].to(torch.float32)
                
                ''' Post-processing '''
                # Trim prompt wav
                wav_pred = wav_pred[vae_latent.size(1)*self.vae_stride*self.hop_size:].cpu().numpy()
                # Norm generated wav to prompt wav's level
                meter = pyln.Meter(self.sr)  # create BS.1770 meter
                loudness_pred = self.loudness_meter.integrated_loudness(wav_pred.astype(float))
                wav_pred = pyln.normalize.loudness(wav_pred, loudness_pred, self.loudness_prompt)
                if np.abs(wav_pred).max() >= 1:
                    wav_pred = wav_pred / np.abs(wav_pred).max() * 0.95

                # Apply hamming window
                wav_pred_.append(wav_pred)

            wav_pred = combine_audio_segments(wav_pred_, sr=self.sr).astype(np.float32)
            waveform = torch.tensor(wav_pred).unsqueeze(0).unsqueeze(0)

            return {"waveform": waveform, "sample_rate": self.sr}

class MegaTTS3SpeakersPreview:
    def __init__(self):
        self.speakers_dir = os.path.join(model_path, "MegaTTS3", "speakers")
    @classmethod
    def INPUT_TYPES(s):
        speakers = get_speakers()
        return {
            "required": {"speaker":(speakers,),},}

    RETURN_TYPES = ("STRING", "AUDIO",)
    RETURN_NAMES = ("speaker", "AUDIO",)
    FUNCTION = "preview"
    CATEGORY = "🎤MW/MW-MegaTTS3"

    def preview(self, speaker):
        wav_path = os.path.join(self.speakers_dir, speaker)
        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.unsqueeze(0)
        output_audio = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        return (wav_path, output_audio,)

INFER_INS_CACHE = None
class MegaTTS3Run:
    def __init__(self):
        self.speakers_dir = os.path.join(model_path, "MegaTTS3", "speakers")
        self.resource_context = None
        self.speaker = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "speaker":("STRING", {"forceInput": True}),
                "text": ("STRING", {"forceInput": True}),
                "text_language": (["en", "zh"], {"default": "zh"}),
                "time_step": ("INT", {"default": 32, "min": 1,}),
                "p_w": ("FLOAT", {"default":1.6, "min": 0.1,}),
                "t_w": ("FLOAT", {"default": 2.5, "min": 0.1,}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "🎤MW/MW-MegaTTS3"

    def clone(self, speaker, text, text_language, time_step, p_w, t_w, unload_model):
        global INFER_INS_CACHE
        if INFER_INS_CACHE is None:
            INFER_INS_CACHE = MegaTTS3DiTInfer()

        latent_file = speaker.replace('.wav', '.npy')

        if os.path.exists(latent_file):

            # 只有音频改变时, 才重新预处理
            if self.speaker is None or self.speaker != speaker:
                self.speaker = speaker
                with open(self.speaker, 'rb') as file:
                    file_content = file.read()
                resource_context = INFER_INS_CACHE.preprocess(file_content, latent_file=latent_file)
                self.resource_context = resource_context
            else:
                resource_context = self.resource_context
        else:
            raise Exception(f"{latent_file}: latent_file not found")
        audio_data = INFER_INS_CACHE.forward(resource_context, text, language_type=text_language, time_step=time_step, p_w=p_w, t_w=t_w)

        if unload_model:
            import gc
            INFER_INS_CACHE.clean()
            INFER_INS_CACHE = None
            self.speaker = None
            self.resource_context = None
            gc.collect()
            torch.cuda.empty_cache()

        return (audio_data,)


class MultiLinePromptMG:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "multi_line_prompt": ("STRING", {
                    "multiline": True, 
                    "default": ""}),
                },
        }

    CATEGORY = "🎤MW/MW-MegaTTS3"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "promptgen"
    
    def promptgen(self, multi_line_prompt: str):
        return (multi_line_prompt.strip(),)


NODE_CLASS_MAPPINGS = {
    "MegaTTS3SpeakersPreview": MegaTTS3SpeakersPreview,
    "MegaTTS3Run": MegaTTS3Run,
    "MultiLinePromptMG": MultiLinePromptMG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaTTS3SpeakersPreview": "MegaTTS3 Speakers Preview",
    "MegaTTS3Run": "Mega TTS3 Run",
    "MultiLinePromptMG": "Multi Line Prompt",
}