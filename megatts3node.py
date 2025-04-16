import json
import os
import io
import librosa
import numpy as np
import torch

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

def get_speakers():
    speakers_dir = os.path.join(model_path, "MegaTTS3", "speakers")
    if not os.path.exists(speakers_dir):
        os.makedirs(speakers_dir, exist_ok=True)
        return []
    
    speakers = [f for f in os.listdir(speakers_dir) if f.endswith('.wav')]
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
        # self.zh_normalizer = ZhNormalizer(overwrite_cache=False, remove_erhua=False, remove_interjections=False)
        # self.en_normalizer = EnNormalizer(overwrite_cache=False)
        
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
        print("▶️ forward函数开始执行...")
        device = self.device
        print(f"▶️ 使用设备: {device}")

        ph_ref = resource_context['ph_ref'].to(device)
        tone_ref = resource_context['tone_ref'].to(device)
        mel2ph_ref = resource_context['mel2ph_ref'].to(device)
        vae_latent = resource_context['vae_latent'].to(device)
        ctx_dur_tokens = resource_context['ctx_dur_tokens'].to(device)
        incremental_state_dur_prompt = resource_context['incremental_state_dur_prompt']
        print("▶️ 资源已加载到设备...")
        
        with torch.inference_mode():
            print("▶️ 开始生成过程...")
            wav_pred_ = []
            # language_type = classify_language(input_text)
            if language_type == 'en':
                # input_text = self.en_normalizer.normalize(input_text)
                text_segs = chunk_text_english(input_text, max_chars=130)  # 使用原始最大长度
                print(f"▶️ 英文文本分段完成，共{len(text_segs)}段")
            else:
                # input_text = self.zh_normalizer.normalize(input_text)
                text_segs = chunk_text_chinese(input_text, limit=60)  # 使用原始最大长度
                print(f"▶️ 中文文本分段完成，共{len(text_segs)}段")

            for seg_i, text in enumerate(text_segs):
                print(f"▶️ 处理第{seg_i+1}/{len(text_segs)}段文本 ({len(text)}字符)")
                ''' G2P '''
                print(f"  ▶️ 执行G2P(grapheme to phoneme)...")
                ph_pred, tone_pred = g2p(self, text)
                print(f"  ✅ G2P完成")

                ''' Duration Prediction '''
                print(f"  ▶️ 执行音素时长预测...")
                mel2ph_pred = dur_pred(self, ctx_dur_tokens, incremental_state_dur_prompt, ph_pred, tone_pred, seg_i, dur_disturb, dur_alpha, is_first=seg_i==0, is_final=seg_i==len(text_segs)-1)
                print(f"  ✅ 音素时长预测完成")
                
                print(f"  ▶️ 准备DiT输入...")
                inputs = prepare_inputs_for_dit(self, mel2ph_ref, mel2ph_pred, ph_ref, tone_ref, ph_pred, tone_pred, vae_latent)
                print(f"  ✅ DiT输入准备完成")
                
                # 使用半精度加速DiT推理
                print(f"  ▶️ 执行DiT推理，timesteps={time_step}...")
                with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                    x = self.dit.inference(inputs, timesteps=time_step, seq_cfg_w=[p_w, t_w]).float()
                print(f"  ✅ DiT推理完成")
                
                # WavVAE decode
                print(f"  ▶️ 执行WavVAE解码...")
                x[:, :vae_latent.size(1)] = vae_latent
                wav_pred = self.wavvae.decode(x)[0,0].to(torch.float32)
                print(f"  ✅ WavVAE解码完成")
                
                ''' Post-processing '''
                print(f"  ▶️ 执行后处理...")
                # Trim prompt wav
                wav_pred = wav_pred[vae_latent.size(1)*self.vae_stride*self.hop_size:].cpu().numpy()
                # Norm generated wav to prompt wav's level
                meter = pyln.Meter(self.sr)  # create BS.1770 meter
                loudness_pred = self.loudness_meter.integrated_loudness(wav_pred.astype(float))
                wav_pred = pyln.normalize.loudness(wav_pred, loudness_pred, self.loudness_prompt)
                if np.abs(wav_pred).max() >= 1:
                    wav_pred = wav_pred / np.abs(wav_pred).max() * 0.95
                print(f"  ✅ 后处理完成")

                # Apply hamming window
                wav_pred_.append(wav_pred)
                print(f"✅ 第{seg_i+1}段处理完成")

            print("▶️ 合并音频片段...")
            wav_pred = combine_audio_segments(wav_pred_, sr=self.sr).astype(np.float32)
            waveform = torch.tensor(wav_pred).unsqueeze(0).unsqueeze(0)
            print(f"✅ 音频生成完成! 长度:{len(wav_pred)/self.sr:.2f}秒")

            return {"waveform": waveform, "sample_rate": self.sr}


# 我们只确保输出一个字符串，而让MegaTTS3Run接受所有类型

class MegaTTS3Run:
    infer_ins_cache = None
    @classmethod
    def INPUT_TYPES(s):
        speakers = get_speakers()
        default_speaker = speakers[0] if speakers else ""
        return {
            "required": {
                "speaker": ("STRING", {"default": default_speaker}),  # 使用STRING类型接受任何字符串
                "text": ("STRING",),
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
        # 确保speaker是字符串类型
        speaker = str(speaker)
        
        # 验证和处理speaker的值
        available_speakers = get_speakers()
        
        # 检查是否是一个已知的speaker文件
        if speaker in available_speakers:
            speaker_name = speaker
        else:
            # 如果是.wav结尾的文件名，尝试直接使用
            if speaker.endswith('.wav') and speaker in available_speakers:
                speaker_name = speaker
            else:
                # 如果输入不是有效的speaker，使用默认的
                print(f"警告: 未知音色 '{speaker}'，使用默认音色")
                speaker_name = available_speakers[0] if available_speakers else ""
                print(f"默认音色: {speaker_name}")
        
        # 构建音频文件路径
        sperker_path = os.path.join(model_path, "MegaTTS3", "speakers", speaker_name)
        print(f"使用音色文件: {sperker_path}")
        
        if MegaTTS3Run.infer_ins_cache is not None:
            infer_ins = MegaTTS3Run.infer_ins_cache
        else:
            infer_ins = MegaTTS3Run.infer_ins_cache = MegaTTS3DiTInfer()
        
        try:
            with open(sperker_path, 'rb') as file:
                file_content = file.read()
        except Exception as e:
            print(f"打开音色文件失败: {e}")
            raise Exception(f"无法打开音色文件: {sperker_path} - {str(e)}")

        latent_file = sperker_path.replace('.wav', '.npy')
        print(f"latent_file: {latent_file}")
        if os.path.exists(latent_file):
            print("开始预处理音频...")
            resource_context = infer_ins.preprocess(file_content, latent_file=latent_file)
            print("预处理完成，开始生成音频...")
        else:
            raise Exception("latent_file not found")
        
        # 使用用户设置的time_step值
        print(f"[信息] 使用设置的time_step={time_step}")
        # 提示用户较高的time_step值会增加处理时间但可能提升情感表达
        if time_step > 16:
            print(f"[提示] 较高的time_step值可能会增加处理时间，但有助于提升情感表达效果")
            
        print(f"开始生成音频... 参数: language={text_language}, time_step={time_step}, p_w={p_w}, t_w={t_w}")
        try:
            audio_data = infer_ins.forward(resource_context, text, language_type=text_language, time_step=time_step, p_w=p_w, t_w=t_w)
            print("音频生成完成!")
        except Exception as e:
            print(f"生成音频时发生错误: {e}")
            raise e

        if unload_model:
            import gc
            infer_ins.clean()
            infer_ins = None
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




class SpeakerPreviewMG:
    @classmethod
    def INPUT_TYPES(s):
        speakers = get_speakers()
        default_speaker = speakers[0] if speakers else ""
        return {
            "required": {
                "speaker":(speakers,{"default": default_speaker}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("原音频", "speaker")
    FUNCTION = "preview_speaker"
    CATEGORY = "🎤MW/MW-MegaTTS3"

    def preview_speaker(self, speaker):
        # 获取音频文件路径
        speaker_path = os.path.join(model_path, "MegaTTS3", "speakers", speaker)
        
        # 加载音频文件
        if os.path.exists(speaker_path):
            # 读取音频文件
            with open(speaker_path, 'rb') as file:
                audio_bytes = file.read()
            
            # 使用librosa加载音频
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            # 转换为ComfyUI可用的格式
            waveform = torch.tensor(audio_data).unsqueeze(0).unsqueeze(0)
            audio_output = {"waveform": waveform, "sample_rate": sr}
            
            print(f"已加载原始音色: {speaker}，采样率: {sr}Hz，长度: {len(audio_data)/sr:.2f}秒")
            return (audio_output, speaker)
        else:
            print(f"错误: 找不到音频文件 {speaker_path}")
            # 返回空音频和speaker名称
            empty_waveform = torch.zeros(1, 1, 100)
            return ({"waveform": empty_waveform, "sample_rate": 24000}, speaker)

NODE_CLASS_MAPPINGS = {
    "MegaTTS3Run": MegaTTS3Run,
    "MultiLinePromptMG": MultiLinePromptMG,
    "SpeakerPreviewMG": SpeakerPreviewMG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaTTS3Run": "Mega TTS3 Run",
    "MultiLinePromptMG": "Multi Line Prompt",
    "SpeakerPreviewMG": "音色预览",
}