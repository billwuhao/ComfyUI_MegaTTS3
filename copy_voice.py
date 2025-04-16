import os
import sys
import shutil

def copy_voice_files(template_wav, new_name):
    """
    复制一个已存在的音色文件(.wav和.npy)并用新名称保存
    """
    if not os.path.exists(template_wav):
        print(f"错误: 模板文件不存在 - {template_wav}")
        return False
        
    # 检查对应的npy文件是否存在
    template_npy = template_wav.replace('.wav', '.npy')
    if not os.path.exists(template_npy):
        print(f"错误: 模板npy文件不存在 - {template_npy}")
        return False
        
    # 获取目标路径
    target_dir = os.path.dirname(template_wav)
    target_wav = os.path.join(target_dir, new_name + '.wav')
    target_npy = os.path.join(target_dir, new_name + '.npy')
    
    # 检查目标文件是否已存在
    if os.path.exists(target_wav):
        print(f"警告: 目标wav文件已存在 - {target_wav}")
        overwrite = input("是否覆盖? (y/n): ")
        if overwrite.lower() != 'y':
            print("操作已取消")
            return False
            
    if os.path.exists(target_npy):
        print(f"警告: 目标npy文件已存在 - {target_npy}")
        overwrite = input("是否覆盖? (y/n): ")
        if overwrite.lower() != 'y':
            print("操作已取消")
            return False
    
    # 复制文件
    try:
        shutil.copy2(template_wav, target_wav)
        shutil.copy2(template_npy, target_npy)
        print(f"✅ 成功复制音色文件:")
        print(f"  WAV: {os.path.basename(template_wav)} → {os.path.basename(target_wav)}")
        print(f"  NPY: {os.path.basename(template_npy)} → {os.path.basename(target_npy)}")
        return True
    except Exception as e:
        print(f"❌ 复制文件失败: {e}")
        return False

if __name__ == "__main__":
    print("\n===== MegaTTS3音色复制工具 =====\n")
    
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("用法: python copy_voice.py <模板wav文件> <新音色名称>")
        print("例如: python copy_voice.py ../../../models/TTS/MegaTTS3/speakers/周杰伦1.wav 我的音色")
        sys.exit(1)
    
    template_wav = sys.argv[1]
    new_name = sys.argv[2]
    
    if copy_voice_files(template_wav, new_name):
        print("\n✅ 音色复制成功!")
        print("您现在可以在ComfyUI的MegaTTS3节点中使用这个新音色了。")
    else:
        print("\n❌ 操作失败")
