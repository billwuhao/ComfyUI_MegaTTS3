[中文](README-CN.md) | [English](README.md)

# MegaTTS3 Voice Cloning Node for ComfyUI

High-quality voice cloning, supports Chinese and English, and can perform cross-lingual cloning.

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/2025-04-06_13-52-57.png)

## 📣 Updates

[2025-04-06]⚒️: Released v1.0.0.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_MegaTTS3.git
cd ComfyUI_MegaTTS3
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

- Models and voices need to be manually downloaded and placed in the `ComfyUI\models\TTS` directory:

[MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3/tree/main) Download the entire folder and place it in the `TTS` folder.

Create a new `speakers` folder inside the `MegaTTS3` folder. Download all `.wav` and `.npy` files from [Google drive](https://drive.google.com/drive/folders/1QhcHWcy20JfqWjgqZX1YM3I6i9u4oNlr) and place them in the `speakers` folder.

![image](https://github.com/billwuhao/ComfyUI_MegaTTS3/blob/main/images/2025-04-06_14-49-12.png)

## Acknowledgements

- [MegaTTS3](https://github.com/bytedance/MegaTTS3)
