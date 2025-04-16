import os
import sys
import torch
import numpy as np
import librosa

# 导入MegaTTS3的模型和工具
from tts.utils.audio_utils.io import convert_to_wav_bytes
from tts.modules.wavvae.decoder.wavvae_v3 import WavVAE_V3  # 使用正确的类名
from tts.utils.commons.hparams import hparams

def extract_latent_from_wav(wav_path, device='cuda'):
    """
    从WAV文件提取特征向量并保存为NPY文件
    使用与MegaTTS3相同的WavVAE编码器
    """
    print(f"正在处理音频: {wav_path}")
    
    # 加载音频文件
    try:
        with open(wav_path, 'rb') as f:
            audio_bytes = f.read()
        print("音频文件加载成功")
    except Exception as e:
        print(f"加载音频失败: {str(e)}")
        return False
    
    # 转换为WAV格式
    wav_bytes = convert_to_wav_bytes(audio_bytes)
    
    # 加载音频数据
    sr = 16000  # MegaTTS3使用的采样率
    try:
        wav, _ = librosa.core.load(wav_bytes, sr=sr)
        print(f"音频加载成功: 采样率={sr}, 时长={len(wav)/sr:.2f}秒")
    except Exception as e:
        print(f"加载音频数据失败: {str(e)}")
        return False
    
    # 应用相同的预处理
    ws = hparams['win_size']
    if len(wav) % ws < ws - 1:
        wav = np.pad(wav, (0, ws - 1 - (len(wav) % ws)), mode='constant', constant_values=0.0).astype(np.float32)
    wav = np.pad(wav, (0, 12000), mode='constant', constant_values=0.0).astype(np.float32)
    
    # 创建WavVAE_V3编码器
    try:
        wavvae = WavVAE_V3(hparams).to(device)
        print("WavVAE_V3编码器创建成功")
    except Exception as e:
        print(f"创建WavVAE_V3编码器失败: {str(e)}")
        print(f"详细错误: {e}")
        return False
    
    # 转换音频为张量
    wav_tensor = torch.FloatTensor(wav)[None].to(device)
    
    # 提取特征
    try:
        with torch.no_grad():
            vae_latent = wavvae.encode_latent(wav_tensor)
            print(f"特征提取成功: shape={vae_latent.shape}")
    except Exception as e:
        print(f"特征提取失败: {str(e)}")
        return False
    
    # 保存为NPY文件
    npy_path = wav_path.replace('.wav', '.npy')
    try:
        np.save(npy_path, vae_latent.cpu().numpy())
        print(f"特征文件保存成功: {npy_path}")
    except Exception as e:
        print(f"保存特征文件失败: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python extract_latent.py 音频文件路径 [设备(cuda/cpu)]")
        sys.exit(1)
    
    wav_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
    
    if not os.path.exists(wav_path):
        print(f"错误: 文件不存在 - {wav_path}")
        sys.exit(1)
    
    # 检查文件扩展名
    if not wav_path.lower().endswith('.wav'):
        print("警告: 文件不是WAV格式，可能会导致处理问题")
    
    # 检查目标NPY文件是否已存在
    npy_path = wav_path.replace('.wav', '.npy')
    if os.path.exists(npy_path):
        print(f"警告: NPY文件已存在 - {npy_path}")
        choice = input("要覆盖现有文件吗? (y/n): ")
        if choice.lower() != 'y':
            print("操作取消")
            sys.exit(0)
    
    print("开始处理...")
    success = extract_latent_from_wav(wav_path, device)
    
    if success:
        print("\n✅ 处理成功!")
        print(f"生成的特征文件: {wav_path.replace('.wav', '.npy')}")
        print("你现在可以在MegaTTS3中使用这个音色了!")
    else:
        print("\n❌ 处理失败!")
        print("请检查上面的错误信息并尝试修复问题")
