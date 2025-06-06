import json
import os
import librosa
import numpy as np
import torch
import torchaudio
from typing import List, Union, Optional
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
from langdetect import detect as classify_language
import pyloudnorm as pyln
import folder_paths
import gc
import re
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator
from tts.frontend_function import g2p, align, make_dur_prompt, dur_pred, prepare_inputs_for_dit
from tts.utils.audio_utils.io import convert_to_wav_bytes, combine_audio_segments
from tts.utils.commons.ckpt_utils import load_ckpt
from tts.utils.commons.hparams import set_hparams, hparams
from tts.utils.text_utils.text_encoder import TokenTextEncoder
from tts.utils.text_utils.split_text import chunk_text_chinese, chunk_text_english, chunk_text_chinesev2
from tts.utils.commons.hparams import hparams, set_hparams


models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")
speakers_dir = os.path.join(model_path, "speakers")
cache_dir = folder_paths.get_temp_directory()

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
    if not os.path.exists(speakers_dir):
        os.makedirs(speakers_dir, exist_ok=True)
        return []
    speakers = get_all_files(speakers_dir, extensions=[".wav", ".mp3", ".flac", ".mp4", ".WAV", ".MP3", ".FLAC", ".MP4"], relative_path=True)
    return speakers


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
        self.zh_normalizer = ZhNormalizer(overwrite_cache=False, remove_erhua=False, remove_interjections=False)
        self.en_normalizer = EnNormalizer(overwrite_cache=False)

        # loudness meter
        self.loudness_meter = pyln.Meter(self.sr)
        
        self.ph_ref = None
        self.tone_ref = None
        self.mel2ph_ref = None
        self.vae_latent = None
        self.ctx_dur_tokens = None
        self.incremental_state_dur_prompt = None

        self.audio_bytes = None
        
    def clean(self):
        import gc
        self.dur_model = None
        self.dit= None
        self.g2p_model = None
        self.wavvae_en = None
        self.wavvae_de = None
        self.aligner_lm = None

        self.audio_bytes = None
        self.ph_ref = None
        self.tone_ref = None
        self.mel2ph_ref = None
        self.vae_latent = None
        self.ctx_dur_tokens = None
        self.incremental_state_dur_prompt = None

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

        self.wavvae_en = WavVAE_V3(hparams=hp_wavvae)
        self.wavvae_de = WavVAE_V3(hparams=hp_wavvae)

        if os.path.exists(f'{self.wavvae_exp_name}/model_only_last.ckpt'):
            load_ckpt(self.wavvae_en, f'{self.wavvae_exp_name}/model_only_last.ckpt', 'model_gen', strict=True)
            self.has_vae_encoder = True
            self.wavvae_en.eval()
            self.wavvae_en.to(device)
        else:
            load_ckpt(self.wavvae_de, f'{self.wavvae_exp_name}/decoder.ckpt', 'model_gen', strict=False)
            self.has_vae_encoder = False
            self.wavvae_de.eval()
            self.wavvae_de.to(device)

        self.vae_stride = hp_wavvae.get('vae_stride', 4)
        self.hop_size = hp_wavvae.get('hop_size', 4)
    
    def preprocess(self, audio_bytes, latent_file=None, topk_dur=1, **kwargs):
        if self.audio_bytes != audio_bytes:
            self.audio_bytes = audio_bytes
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

            self.kv_cache = None
            self.hooks = None

            with torch.inference_mode():
                ''' Forward WaveVAE to obtain: prompt latent '''
                if self.has_vae_encoder:
                    if latent_file is None:
                        wav = torch.FloatTensor(wav)[None].to(self.device)
                        vae_latent = self.wavvae_en.encode_latent(wav)
                    else:
                        vae_latent = torch.from_numpy(np.load(latent_file)).to(self.device)
                    vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
                else:
                    assert latent_file is not None, "WaveVAE encode model does not exist, an npy file must be provided!!!"
                    vae_latent = torch.from_numpy(np.load(latent_file)).to(self.device)
                    vae_latent = vae_latent[:, :mel2ph_ref.size(1)//4]
            
                ''' Duration Prompting '''
                self.dur_model.hparams["infer_top_k"] = topk_dur if topk_dur > 1 else None
                incremental_state_dur_prompt, ctx_dur_tokens = make_dur_prompt(self, mel2ph_ref, ph_ref, tone_ref)

                self.ph_ref = ph_ref.to(self.device)
                self.tone_ref = tone_ref.to(self.device)
                self.mel2ph_ref = mel2ph_ref.to(self.device)
                self.vae_latent = vae_latent.to(self.device)
                self.ctx_dur_tokens = ctx_dur_tokens.to(self.device)
                self.incremental_state_dur_prompt = incremental_state_dur_prompt

    def forward(self, texts, time_step, p_w, t_w, dur_disturb=0.1, dur_alpha=1.0, **kwargs):

        with torch.inference_mode():
            ''' Generating '''
            waveforms = []
            for input_text in texts:
                wav_pred_ = []
                language_type = classify_language(input_text)
                if language_type == 'en':
                    input_text = self.en_normalizer.normalize(input_text)
                    text_segs = chunk_text_english(input_text, max_chars=130)
                else:
                    input_text = self.zh_normalizer.normalize(input_text)
                    text_segs = chunk_text_chinesev2(input_text, limit=60)

                for seg_i, text in enumerate(text_segs):
                    ''' G2P '''
                    ph_pred, tone_pred = g2p(self, text)

                    ''' Duration Prediction '''
                    mel2ph_pred = dur_pred(self, self.ctx_dur_tokens, self.incremental_state_dur_prompt, ph_pred, tone_pred, seg_i, dur_disturb, dur_alpha, is_first=seg_i==0, is_final=seg_i==len(text_segs)-1)
                    
                    inputs = prepare_inputs_for_dit(self, self.mel2ph_ref, mel2ph_pred, self.ph_ref, self.tone_ref, ph_pred, tone_pred, self.vae_latent)
                    # Speech dit inference
                    with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                        x = self.dit.inference(inputs, timesteps=time_step, seq_cfg_w=[p_w, t_w]).float()
                    
                    # WavVAE decode
                    x[:, :self.vae_latent.size(1)] = self.vae_latent
                    if self.has_vae_encoder:
                        wav_pred = self.wavvae_en.decode(x)[0,0].to(torch.float32)
                    else:
                        wav_pred = self.wavvae_de.decode(x)[0,0].to(torch.float32)
                    
                    ''' Post-processing '''
                    # Trim prompt wav
                    wav_pred = wav_pred[self.vae_latent.size(1)*self.vae_stride*self.hop_size:].cpu().numpy()
                    # Norm generated wav to prompt wav's level
                    meter = pyln.Meter(self.sr)  # create BS.1770 meter
                    loudness_pred = self.loudness_meter.integrated_loudness(wav_pred.astype(float))
                    wav_pred = pyln.normalize.loudness(wav_pred, loudness_pred, self.loudness_prompt)
                    if np.abs(wav_pred).max() >= 1:
                        wav_pred = wav_pred / np.abs(wav_pred).max() * 0.95

                    # Apply hamming window
                    wav_pred_.append(wav_pred)

                    gc.collect()
                    torch.cuda.empty_cache()

                wav_pred = combine_audio_segments(wav_pred_, sr=self.sr).astype(np.float32)
                waveform = torch.tensor(wav_pred)
                waveforms.append(waveform.cpu())

            return torch.cat(waveforms, dim=0), self.sr


class MegaTTS3SpeakersPreview:
    @classmethod
    def INPUT_TYPES(s):
        speakers = get_speakers()
        return {
            "required": {"speaker":(speakers,),},}

    RETURN_TYPES = ("AUDIO", "STRING", )
    RETURN_NAMES = ("audio", "npy_file", )
    FUNCTION = "preview"
    CATEGORY = "🎤MW/MW-MegaTTS3"

    def preview(self, speaker):
        wav_path = os.path.join(speakers_dir, speaker)
        latent_file = wav_path.rsplit('.', 1)[0] + '.npy'
        if not os.path.exists(latent_file):
            latent_file = ""

        waveform, sample_rate = torchaudio.load(wav_path)
        waveform = waveform.unsqueeze(0)
        output_audio = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        return (output_audio, latent_file)


def cache_audio_tensor(
    cache_dir,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")

def statistical_compare(tensor1, tensor2):
    """通过统计特征快速比较"""
    stats1 = {
        'mean': tensor1.mean(),
        'std': tensor1.std(),
        'max': tensor1.max(),
        'min': tensor1.min()
    }
    stats2 = {
        'mean': tensor2.mean(),
        'std': tensor2.std(),
        'max': tensor2.max(),
        'min': tensor2.min()
    }
    return all(torch.allclose(stats1[k], stats2[k], rtol=1e-3) for k in stats1)


INFER_INS_CACHE = None
class MegaTTS3Run:
    def __init__(self):
        self.resource_context = None
        self.audio_tensor = None
        self.audio_prompt = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("STRING", {"forceInput": True}),
                "time_step": ("INT", {"default": 32, "min": 1,}),
                "p_w": ("FLOAT", {"default":1.6, "min": 0.1,}),
                "t_w": ("FLOAT", {"default": 2.5, "min": 0.1,}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "dialogue_audio_s2":("AUDIO",),
                "audio_npy_file": ("STRING",  {"forceInput": True, "tooltip": "No `npy_file` will use VAE to encode audio. 不提供 .npy 文件, 将使用 WaveVAE 编码音频"}),
                "audio_s2_npy_file": ("STRING",  {"forceInput": True, "tooltip": "No `npy_file` will use VAE to encode audio. 不提供 .npy 文件, 将使用 WaveVAE 编码音频"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "🎤MW/MW-MegaTTS3"

    def clone(self, audio, text, time_step, p_w, t_w, unload_model, audio_npy_file=None, dialogue_audio_s2=None, audio_s2_npy_file=None):
        if not os.path.exists(os.path.join(model_path, "MegaTTS3", 'wavvae', 'model_only_last.ckpt')):
            print("WaveVAE encode model does not exist, an npy file must be provided!!!")
        waveform = audio["waveform"].squeeze(0)

        global INFER_INS_CACHE
        if INFER_INS_CACHE is None:
            INFER_INS_CACHE = MegaTTS3DiTInfer()
            
        latent_file = audio_npy_file if audio_npy_file else None
        try:
            import gc
            if dialogue_audio_s2 is None:
                # 只有音频改变时, 才重新预处理
                if self.audio_tensor is None or self.audio_prompt is None or statistical_compare(self.audio_tensor, waveform) == False:
                    self.audio_tensor = waveform
                    self.audio_prompt = cache_audio_tensor(cache_dir, waveform, audio["sample_rate"])

                texts = [i.strip() for i in re.split(r'\n\s*\n', text.strip()) if i.strip()]
                with open(self.audio_prompt, 'rb') as file:
                    file_content = file.read()
                INFER_INS_CACHE.preprocess(file_content, latent_file=latent_file)

                del file_content
                gc.collect()
                torch.cuda.empty_cache()

                waveform, sr = INFER_INS_CACHE.forward(texts=texts, time_step=time_step, p_w=p_w, t_w=t_w)
                gc.collect()
                torch.cuda.empty_cache()
            else:
                latent_file_2 = audio_s2_npy_file if audio_s2_npy_file else None
                audio_1 = cache_audio_tensor(cache_dir, waveform, audio["sample_rate"])
                audio_2 = cache_audio_tensor(cache_dir, dialogue_audio_s2["waveform"].squeeze(0), dialogue_audio_s2["sample_rate"])
                with open(audio_1, 'rb') as file:
                    file_content_1 = file.read()
                with open(audio_2, 'rb') as file:
                    file_content_2 = file.read()

                gc.collect()
                torch.cuda.empty_cache()
        
                ress = []
                for t, a, n in self.get_speaker_text_audio(text, audio_1, audio_2):
                    texts = [i.strip() for i in re.split(r'\n\s*\n', t.strip()) if i.strip()]
                    if a == audio_1:
                        INFER_INS_CACHE.preprocess(file_content_1, latent_file=latent_file)
                        res_sub, sr = INFER_INS_CACHE.forward(texts=texts, time_step=time_step, p_w=p_w, t_w=t_w)
                        ress.append([res_sub, n])
                    else:
                        INFER_INS_CACHE.preprocess(file_content_2, latent_file=latent_file_2)
                        res_sub, sr = INFER_INS_CACHE.forward(texts=texts, time_step=time_step, p_w=p_w, t_w=t_w)
                        ress.append([res_sub, n])

                del file_content_1
                del file_content_2
                gc.collect()
                torch.cuda.empty_cache()
                waveform = torch.cat(list(zip(*sorted(ress, key=lambda x: x[1])))[0], dim=0)

        except Exception as e:
            if unload_model:
                import gc
                INFER_INS_CACHE.clean()
                INFER_INS_CACHE = None
                self.resource_context = None
                gc.collect()
                torch.cuda.empty_cache()
            raise e

        if unload_model:
            import gc
            INFER_INS_CACHE.clean()
            INFER_INS_CACHE = None
            self.resource_context = None
            gc.collect()
            torch.cuda.empty_cache()

        return ({"waveform": waveform.unsqueeze(0).unsqueeze(0), "sample_rate": sr},)

    def get_speaker_text_audio(self, text, audio_1, audio_2):
        pattern = r'(\[s?S?1\]|\[s?S?2\])\s*([\s\S]*?)(?=\[s?S?[12]\]|$)'
        matches = re.findall(pattern, text)
        if len(matches) == 0:
            raise ValueError("No speaker tags found in the text: [S1]... [S2]...")
        labels = []
        contents = []
        audios = []

        for label, content in matches:
            labels.append(label)
            contents.append(content)
    
        audios = [
            audio_1 if i.lower() == '[s1]' else audio_2 for i in labels
        ]

        return sorted(zip(contents, audios, range(len(contents))), key=lambda x: x[1])
    

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
    RETURN_NAMES = ("text",)
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
    "MegaTTS3Run": "MegaTTS3 Run",
    "MultiLinePromptMG": "Multi Line Text",
}