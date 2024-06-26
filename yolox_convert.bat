@ ECHO off
set MODEL_SRC_DIR=C:\ProgramData\miniforge3
set CONDA_ENV=yolox_ti

set YOLOX_DIR=%~dp0

set MODEL_FILE_NAME=yolox_nano_ti_lite_nu_hg_150
set YOLOX_M_CONFIG=exps/default/yolox_nano_ti_lite_nu_hg.py
set OUTPUT_DIR=YOLOX_outputs/yolox_nano_ti_lite_nu_hg_150
set TRAIN_DATASET=datasets\hagrid_coco\train2017

set YOLOX_PYTORCH=%OUTPUT_DIR%/latest_ckpt.pth
set YOLOX_ONNX=%OUTPUT_DIR%/%MODEL_FILE_NAME%.onnx
set CALIB_DATA=%OUTPUT_DIR%/calib_data_320x320_n200_hg.npy

call %MODEL_SRC_DIR%\Scripts\activate.bat
call conda activate %CONDA_ENV%
::call cd %~dp0
call cd %YOLOX_DIR%

::pytorch => onnx => tflite
@echo on
call python tools/export_onnx.py -f %YOLOX_M_CONFIG% -c %YOLOX_PYTORCH% --output-name %YOLOX_ONNX%
call python demo/TFLite/generate_calib_data.py --img-size 320 320 --n-img 200 -o %CALIB_DATA% --img-dir %TRAIN_DATASET%
call onnx2tf -i %YOLOX_ONNX% -oiqt -qcind images %CALIB_DATA% "[[[[0,0,0]]]]" "[[[[1,1,1]]]]"
call robocopy saved_model %OUTPUT_DIR% /move

::vela
@echo off
set MODEL_TFLITE_FILE=%MODEL_FILE_NAME%_full_integer_quant.tflite
set MODEL_OPTIMISE_FILE=%MODEL_FILE_NAME%_full_integer_quant_vela.tflite
set VELA_OUTPUT_DIR=%OUTPUT_DIR%\vela
set TEMPLATES_DIR=vela\Tool\tflite2cpp\templates
::accelerator config. ethos-u55-32, ethos-u55-64, ethos-u55-128, ethos-u55-256, ethos-u65-256, ethos-u65-512
set VELA_ACCEL_CONFIG=ethos-u55-256
::optimise option. Size, Performance
set VELA_OPTIMISE_OPTION=Performance
::configuration file
set VELA_CONFIG_FILE=vela\Tool\vela\default_vela.ini
::memory mode. Selects the memory mode to use as specified in the vela configuration file
set VELA_MEM_MODE=Shared_Sram
::system config. Selects the system configuration to use as specified in the vela configuration file
set VELA_SYS_CONFIG=Ethos_U55_High_End_Embedded

set vela_argu= %OUTPUT_DIR%\%MODEL_TFLITE_FILE% --accelerator-config=%VELA_ACCEL_CONFIG% --optimise %VELA_OPTIMISE_OPTION% --config %VELA_CONFIG_FILE% --memory-mode=%VELA_MEM_MODE% --system-config=%VELA_SYS_CONFIG% --output-dir=%VELA_OUTPUT_DIR%
set model_argu= --tflite_path %VELA_OUTPUT_DIR%\%MODEL_OPTIMISE_FILE% --output_dir %VELA_OUTPUT_DIR% --template_dir %TEMPLATES_DIR%

if not exist "%VELA_OUTPUT_DIR%" (
    echo Folder does not exist. Creating folder...
    mkdir "%VELA_OUTPUT_DIR%"
    echo Folder created successfully.
) else (
    echo Folder already exists.
)

@echo on
vela\Tool\vela\vela-3_10_0.exe %vela_argu%
vela\Tool\tflite2cpp\gen_model_cpp.exe %model_argu%

pause