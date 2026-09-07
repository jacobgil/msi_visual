@echo off
REM Example usage scripts for train_parametric_mics_lmc.py (Windows)

REM Basic training on a single file
python train_parametric_mics_lmc.py data.input_path=C:\path\to\image.npy

REM Training on a folder of files
python train_parametric_mics_lmc.py data.input_path=C:\path\to\folder\

REM Training with custom output directory
python train_parametric_mics_lmc.py data.input_path=C:\path\to\image.npy output_dir=C:\path\to\outputs

REM Sweep over sampling methods
python train_parametric_mics_lmc.py --multirun data.input_path=C:\path\to\image.npy model.sampling=random,coreset,kmeans++,kmeans

REM Sweep over multiple hyperparameters
python train_parametric_mics_lmc.py --multirun data.input_path=C:\path\to\image.npy model.sampling=random,coreset model.num_epochs=50,100,200 model.beta=0.0,1.0,5.0

REM Training with verbose output
python train_parametric_mics_lmc.py data.input_path=C:\path\to\image.npy model.verbose=true

REM Training with clustering enabled
python train_parametric_mics_lmc.py data.input_path=C:\path\to\image.npy model.cluster=true model.clusters=[32,64,128,256,512] model.beta=1.0

REM Training with immediate inference
python train_parametric_mics_lmc.py data.input_path=C:\path\to\training\folder\ inference.inference_folder=C:\path\to\inference\folder\

REM Training with immediate inference using glob pattern
python train_parametric_mics_lmc.py data.input_path=C:\path\to\training\folder\ inference.inference_folder=C:\path\to\inference\folder\*_test.npy

REM Inference mode: Run inference on folder of files
python train_parametric_mics_lmc.py inference.model_path=C:\path\to\trained_model.pth inference.inference_folder=C:\path\to\folder\

REM Inference mode: Run inference with glob pattern
python train_parametric_mics_lmc.py inference.model_path=C:\path\to\trained_model.pth inference.inference_folder=C:\path\to\folder\*.npy

REM Inference mode: Run inference with specific pattern
python train_parametric_mics_lmc.py inference.model_path=C:\path\to\trained_model.pth inference.inference_folder=C:\path\to\folder\*_test.npy

