# YOLOX-ti-lite-nu tflite version for MCU with/wo NPU device
 This repository is a fork of [TexasInstruments/edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox) and [motokimura/yolox-ti-lite_tflite](https://github.com/motokimura/yolox-ti-lite_tflite).

 - add `exps\default\yolox_nano_ti_lite_nu.py` for 320X320 depthwise YOLOX-ti-lite version.
 - Add int8/f32 tflite mAP evaluating script: `demo\TFLite\tflite_eval` 
 - support models: tflite_yolox_nano_ti, tflite_yolofastest_v1(mAP is lower)
 - Installation:
 - How to Use:
 - You can also reference the original [readme](https://github.com/MaxCYCHEN/yolox-ti-lite_tflite_int8/blob/main/README_motokimura.md) or [edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)

 ## Installation
 - Create a new python env. If you aren't familiar with python env creating, you can reference here: [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise?tab=readme-ov-file#2-installation--env-create)
 - upgrade pip
 ```bash 
pip3 install --upgrade pip setuptools
```

- 1. Installing pytorch, basing on the type of system, CUDA version, PyTorch version, and MMCV version [pytorch_locally](https://pytorch.org/get-started/locally/)
```bash 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- 2. Installing mmcv, basing on your hardware config [Install with pip](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip)
```bash 
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

- 3. Installing other python requirements
```bash
pip3 install --no-input -r requirements.txt
```

 ## How to Use

