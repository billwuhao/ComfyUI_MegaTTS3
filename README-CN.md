[中文](README-CN.md) | [English](README.md) 

# ComfyUI 的 MegaTTS3 声音克隆节点

声音克隆质量非常高, 支持中英文, 并可跨语言克隆. **支持自定义音色!!! 超长文本!!! 双人对话!!! Windows 正常安装 pynini, 不再是阉割版 TTS!!!**.

## 📣 更新

[2025-06-07]⚒️: v2.0.0. **支持自定义音色, 支持超长文本, 支持双人对话, Windows 正常安装 pynini, 不再是阉割版 TTS!**.

```
[S1] MegaTTS 真开源版本来了，效果666
[S2] 晕 xuan4 是一种 gan3 觉
[S1] 我爱你！I love you!“我爱你”的英语是“I love you”
[S2] 2.5平方电线,共465篇，约315万字
[S1] 2002年的第一场雪，下在了2003年
```

[2025-04-28]⚒️: 新增预览音色节点, 先预览音色, 满意再进行克隆. 感谢 @chenpipi0807 的 idea😍. 可在 `speakers` 文件夹下分门别类建更多文件夹.

[2025-04-06]⚒️: 发布 v1.0.0.

## 使用

- 单人克隆(超长文本用空行隔开):

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/2025-04-06_13-52-57.png)

- 双人对话:

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/2025-04-06_14-49-12.png)

## 安装

- **Windows 先安装以下依赖**:

[pynini-windows-wheels](https://github.com/billwuhao/pynini-windows-wheels/releases/tag/v2.1.6.post1) 下载相应 python 版本的 pynini 轮子.

示例:
```
D:\AIGC\python\py310\python.exe -m pip install pynini-2.1.6.post1-cp3xx-cp3xx-win_amd64.whl
D:\AIGC\python\py310\python.exe -m pip install importlib_resources
D:\AIGC\python\py310\python.exe -m pip install WeTextProcessing>=1.0.4 --no-deps
```

- **然后正常进行下列安装**:
```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_MegaTTS3.git
cd ComfyUI_MegaTTS3
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

- 模型和音色需要手动下载放到 `ComfyUI\models\TTS` 路径下:

[MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3/tree/main)  整个文件夹全部下载放到 `TTS` 文件夹下.

- **VAE 编码模型, 加微信公众号获取, 放到 `TTS\MegaTTS3\wavvae` 文件夹下, 即可自定义音色而无需 `.npy` 文件**:

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/gzh.webp)

- 请将音频放到 `TTS\speakers` 目录下. 我将会把所有 TTS 节点的说话者音频全部统一放到 `ComfyUI\models\TTS\speakers` 路径下, 这些节点包括 `IndexTTS, CSM, Dia, KokoroTTS, MegaTTS, QuteTTS, SparkTTS, StepAudioTTS` 等.

结构如下:

```
.
│  .gitattributes
│  config.json
│  README.md
│
├─aligner_lm
│      config.yaml
│      model_only_last.ckpt
│
├─diffusion_transformer
│      config.yaml
│      model_only_last.ckpt
│
├─duration_lm
│      config.yaml
│      model_only_last.ckpt
│
├─g2p
│      added_tokens.json
│      config.json
│      generation_config.json
│      latest
│      merges.txt
│      model.safetensors
│      special_tokens_map.json
│      tokenizer.json
│      tokenizer_config.json
│      trainer_state.json
│      vocab.json
│
└─wavvae
        config.yaml
        decoder.ckpt
        model_only_last.ckpt
```

## 鸣谢

- [MegaTTS3](https://github.com/bytedance/MegaTTS3)

## 打赏

您的赞赏是我最大的动力! 感谢您支持我一杯咖啡!

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/20250607012102.jpg)