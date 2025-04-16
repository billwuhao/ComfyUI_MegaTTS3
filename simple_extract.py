import os
import sys
import torch
import numpy as np
from glob import glob

def create_npy_from_wav(wav_path, template_npy_path=None):
    """
    从WAV文件创建NPY文件，使用已有的NPY文件作为模板
    """
    print(f"正在处理: {wav_path}")
    
    # 检查WAV文件是否存在
    if not os.path.exists(wav_path):
        print(f"错误: WAV文件不存在 - {wav_path}")
        return False
    
    # 生成NPY文件路径
    npy_path = wav_path.replace('.wav', '.npy')
    
    # 如果没有提供模板文件，则查找speaker目录中的第一个NPY文件作为模板
    if template_npy_path is None:
        speaker_dir = os.path.dirname(wav_path)
        template_files = glob(os.path.join(speaker_dir, "*.npy"))
        if template_files:
            template_npy_path = template_files[0]
            print(f"使用模板文件: {template_npy_path}")
        else:
            print("错误: 找不到模板NPY文件，请确保speaker目录中有至少一个.npy文件")
            return False
    
    # 复制模板文件的结构，但使用随机值
    try:
        # 加载模板NPY
        template_data = np.load(template_npy_path)
        print(f"模板形状: {template_data.shape}")
        
        # 创建相同形状的新数组
        new_data = np.random.randn(*template_data.shape).astype(template_data.dtype)
        
        # 保存到新的NPY文件
        np.save(npy_path, new_data)
        print(f"已创建占位符NPY文件: {npy_path}")
        print("注意: 这是一个随机生成的占位符文件，可能不会产生理想的音频效果")
        return True
    except Exception as e:
        print(f"处理失败: {e}")
        return False

if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) < 2:
        print("用法: python simple_extract.py <wav文件路径> [模板npy文件路径]")
        sys.exit(1)
    
    wav_path = sys.argv[1]
    template_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("\n===== 简易NPY文件生成器 =====")
    print("这个脚本会创建一个与模板结构相同的NPY文件")
    print("注意: 生成的文件是随机值，只是为了让系统能加载，但音色效果可能不理想")
    print("=============================\n")
    
    if create_npy_from_wav(wav_path, template_path):
        print("\n✅ 完成!")
        print(f"NPY文件已创建: {wav_path.replace('.wav', '.npy')}")
        print("现在您可以在ComfyUI中尝试使用这个音色，但请注意音质可能不理想")
        print("如果您需要更好的结果，请考虑使用官方的提取工具")
    else:
        print("\n❌ 失败")
        print("请检查错误信息并尝试解决问题")
