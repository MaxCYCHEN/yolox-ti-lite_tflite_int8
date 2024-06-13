@ ECHO off
set MODEL_SRC_DIR=C:\ProgramData\miniforge3
set CONDA_ENV=yolox_ti

set YOLOX_DIR=C:\Users\USER\Desktop\ML\yolox-ti-lite_tflite_int8

set YOLOX_M_CONFIG=exps/default/yolox_nano_ti_lite_nu.py
set OUTPUT_DIR=YOLOX_outputs/yolox_nano_ti_lite_nu_medicinev2_rgb
set YOLOX_PYTORCH=%OUTPUT_DIR%/latest_ckpt.pth
set YOLOX_ONNX=%OUTPUT_DIR%/yolox_nano_nu_medicinev2_rgb.onnx
set CALIB_DATA=%OUTPUT_DIR%/calib_data_320x320_n200_hg.npy
set TRAIN_DATASET=datasets\medicinev2_coco\train2017

call %MODEL_SRC_DIR%\Scripts\activate.bat

call conda activate %CONDA_ENV%

::call cd %~dp0
call cd %YOLOX_DIR%

@echo on
call python tools/export_onnx.py -f %YOLOX_M_CONFIG% -c %YOLOX_PYTORCH% --output-name %YOLOX_ONNX%

call python demo/TFLite/generate_calib_data.py --img-size 320 320 --n-img 200 -o %CALIB_DATA% --img-dir %TRAIN_DATASET%

call onnx2tf -i %YOLOX_ONNX% -oiqt -qcind images %CALIB_DATA% "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"

call robocopy saved_model %OUTPUT_DIR% /move


pause