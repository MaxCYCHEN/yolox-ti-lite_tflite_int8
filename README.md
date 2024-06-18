# YOLOX-ti-lite-nu tflite version for MCU with/wo NPU device
This repository focuses on YOLOX Nano with slight updates, which can be deployed on Nuvoton devices.   

 - add `exps\default\yolox_nano_ti_lite_nu.py` for 320X320 depthwise YOLOX-ti-lite version.
 - Add int8/f32 tflite mAP evaluating script: `demo\TFLite\tflite_eval`. Support models: tflite_yolox_nano_ti, tflite_yolofastest_v1(mAP is lower)
 - Installation:
 - How to Use:
 - This repository is a fork of [TexasInstruments/edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox) and [motokimura/yolox-ti-lite_tflite](https://github.com/motokimura/yolox-ti-lite_tflite).
 - You can also reference the original [readme](https://github.com/MaxCYCHEN/yolox-ti-lite_tflite_int8/blob/main/README_motokimura.md) or [edgeai-yolox](https://github.com/TexasInstruments/edgeai-yolox)

 ## Installation
 - Create a new python env. If you aren't familiar with python env creating, you can reference here: [NuEdgeWise](https://github.com/OpenNuvoton/NuEdgeWise?tab=readme-ov-file#2-installation--env-create)
 - upgrade pip
 ```bash 
pip3 install --upgrade pip setuptools
```

**1.** Installing pytorch, basing on the type of system, CUDA version, PyTorch version, and MMCV version [pytorch_locally](https://pytorch.org/get-started/locally/)
```bash 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2.** Installing mmcv, basing on your hardware config [Install with pip](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip)
```bash 
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

**3.** Installing other python requirements
```bash
pip3 install --no-input -r requirements.txt
```

 ## How to Use
 #### Train
 ```bash
python tools/train.py -f <MODEL CONFIG FILE> -d 1 -b <BATCH SIZE> --fp16 -o -c <PRETRAIN MODEL PATH>
```
- example:
```bash
python tools/train.py -f exps/default/yolox_nano_ti_lite_nu.py -d 1 -b 64 --fp16 -o -c pretrain/tflite_yolox_nano_ti/320_DW/yolox_nano_320_DW_ti_lite.pth
```
- Please use `exps/default/yolox_nano_ti_lite_nu.py` for yolox-nano-ti-nu model which is able running on Nuvoton devices.
- Custome Train:
    - dataset format: COCO json format
    - The dataset structure and folders name must same as below:
    ```bash
    <dataset_name>
          |
          |----annotations
          |        |----------<train_annotation_json_file>
          |        |----------<val_annotation_json_file>
          |
          |----train2017
          |        |---------train img
          |
          |----val2017
                  |---------validation img
    ```
    - update dataset path
    ```bash
    self.data_dir = "datasets/hagrid_coco"
    self.train_ann = "hagrid_train.json"
    self.val_ann = "hagrid_val.json" 
    ```
