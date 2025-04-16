

# MegaTTS3 语音克隆节点 (ComfyUI版)

高质量语音克隆，支持中文和英文，相似度非常高非常好玩。
![微信截图_20250416194242](https://github.com/user-attachments/assets/25341a4f-a300-4318-953e-f45d8d3fe150)
![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/2025-04-06_13-52-57.png)
![微信截图_20250408171024](https://github.com/user-attachments/assets/47f01ba1-cd9a-4510-b199-bc75ce2cc4bd)

## 📣 音色模型相关说明

音色模型委托官方训练：https://drive.google.com/drive/folders/1gCWL1y_2xu9nIFhUX_OW5MbcFuB7J5Cl
音色模型下载：https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr
格式要求： 所有音频必须是.wav格式且时长小于24秒，文件名中不能包含空格。


## 📣 更新日志

[2025-04-16]🔄: 添加了以下功能：
- 增加「音色预览」节点，可直接预览原始音色并连接到主节点
- 移除time_step参数的自动限制，现在可以使用更高的值（如32）提升情感表达
- 优化speaker参数处理逻辑，改进错误处理

[2025-04-06]⚒️: 发布 v1.0.0 版本

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/chenpipi0807/ComfyUI_MegaTTS3_PIP.git
cd ComfyUI_MegaTTS3_PIP
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

- 模型和音色需要手动下载并放置到 `ComfyUI\models\TTS` 目录中：

从 [MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3/tree/main) 下载整个文件夹并放置到 `TTS` 文件夹中。

在 `MegaTTS3` 文件夹内创建新的 `speakers` 文件夹。从 [Google drive](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr) 下载所有 `.wav` 和 `.npy` 文件，并放置到 `ComfyUI\models\TTS\speakers` 文件夹中。

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/2025-04-06_14-49-12.png)

## 使用说明

### 基本用法
- 添加 `Mega TTS3 Run` 节点并连接文本输入
- 从下拉菜单中选择音色
- 调整参数（time_step、p_w、t_w）以优化输出质量

### 音色预览节点
- 添加 `音色预览` 节点可预览原始音色
- 将 `speaker` 输出直接连接到 `Mega TTS3 Run` 节点的 `speaker` 输入
- 将 `原音频` 输出连接到音频播放器节点进行预览

### 示例工作流
我提供了一个示例工作流 `workflow/TTS3-PIP.json`，展示了：
- 音色预览功能的使用方法
- 音色预览节点与 TTS 节点的连接
- 文本输入和音频生成的完整流程

使用方法：在 ComfyUI 中选择“打开”，加载这个工作流文件即可。

### 性能调优
- 增加 `time_step` 值（如设为32）可提高情感表达质量，但会增加处理时间
- 调整 `p_w`（个性权重）和 `t_w`（文本内容权重）可平衡声音特性和内容表达

## 致谢

- [MegaTTS3](https://github.com/bytedance/MegaTTS3)
