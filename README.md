[中文](README-CN.md) | [English](README.md)

# MegaTTS3 Voice Cloning Nodes for ComfyUI

High-quality voice cloning, supporting both Chinese and English, with cross-lingual cloning capabilities. **Supports custom voice cloning!!! Extra-long text!!! Two-person dialogue!!! Full pynini installation on Windows, no more stripped-down TTS!!!**.

## 📣 Updates

[2025-06-07]⚒️: v2.0.0. **Supports custom voice cloning, extra-long text, two-person dialogue, and full pynini installation on Windows, no more stripped-down TTS!**.

```
[S1] MegaTTS 真开源版本来了，效果666
[S2] 晕 xuan4 是一种 gan3 觉
[S1] 我爱你！I love you!“我爱你”的英语是“I love you”
[S2] 2.5平方电线,共465篇，约315万字
[S1] 2002年的第一场雪，下在了2003年
```


[2025-04-28]⚒️: Added a voice preview node. Preview the voice first, then clone if you're satisfied. Thanks to @chenpipi0807 for the idea😍. You can create categorized subfolders within the `speakers` folder.

[2025-04-06]⚒️: Released v1.0.0.

## Usage

- Single-person cloning (separate long text with blank lines):

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/2025-04-06_13-52-57.png)

- Two-person dialogue:

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/2025-04-06_14-49-12.png)

## Installation

- **For Windows, install the following dependencies first**:

[pynini-windows-wheels](https://github.com/billwuhao/pynini-windows-wheels/releases/tag/v2.1.6.post1) Download the pynini wheel file corresponding to your Python version.

Example:
```
D:\AIGC\python\py310\python.exe -m pip install pynini-2.1.6.post1-cp3xx-cp3xx-win_amd64.whl
D:\AIGC\python\py310\python.exe -m pip install importlib_resources
D:\AIGC\python\py310\python.exe -m pip install WeTextProcessing>=1.0.4 --no-deps
```

- **Then, proceed with the normal installation**:
```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_MegaTTS3.git
cd ComfyUI_MegaTTS3
pip install -r requirements.txt

# For python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

- Models and voices need to be downloaded manually and placed in the `ComfyUI\models\TTS` directory:

[MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3/tree/main) Download the entire folder and place it in the `TTS` directory.

- **For the VAE encoder model, which enables custom voice cloning without `.npy` files, please follow our WeChat Official Account to obtain it. Place it in the `TTS\MegaTTS3\wavvae` folder**:

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/gzh.webp)

- Please place the audio in the `TTS\speakers` directory. I will unify all speaker audios for TTS nodes into the `ComfyUI\models\TTS\speakers` path. These nodes include `IndexTTS, CSM, Dia, KokoroTTS, MegaTTS, QuteTTS, SparkTTS, StepAudioTTS`, etc.

The structure is as follows:

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


## Credits

- [MegaTTS3](https://github.com/bytedance/MegaTTS3)

## Donation

Your appreciation is my greatest motivation! Thank you for supporting me with a cup of coffee!

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/20250607012102.jpg)