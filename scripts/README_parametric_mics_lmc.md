# Training Script for MSIParametricMiCSLMC

This script provides a command-line interface for training MSIParametricMiCSLMC models with Hydra configuration management and hyperparameter sweeps.

## Installation

Make sure you have the required dependencies:

```bash
pip install hydra-core omegaconf
```

## Basic Usage

### Training Mode

#### Single File Training

Train on a single .npy file:

```bash
cd scripts
python train_parametric_mics_lmc.py data.input_path=/path/to/image.npy
```

#### Folder Training

Train on all .npy files in a folder:

```bash
python train_parametric_mics_lmc.py data.input_path=/path/to/folder/
```

### Training with Immediate Inference

Train a model and immediately run inference on a different folder:

```bash
python train_parametric_mics_lmc.py \
    data.input_path=/path/to/training/folder/ \
    inference.inference_folder=/path/to/inference/folder/
```

This will:
1. Train the model on files in the training folder
2. Automatically run inference on files in the inference folder using the trained model
3. Save both training results and inference visualizations to the output directory

### Inference Mode

#### Run Inference on Folder of Files

Load a trained model and generate visualizations for all .npy files in a folder:

```bash
python train_parametric_mics_lmc.py \
    inference.model_path=/path/to/trained_model.pth \
    inference.inference_folder=/path/to/folder/
```

#### Run Inference with Glob Pattern

Use glob patterns to select specific files:

```bash
# All .npy files in folder
python train_parametric_mics_lmc.py \
    inference.model_path=/path/to/trained_model.pth \
    inference.inference_folder=/path/to/folder/*.npy

# Specific pattern
python train_parametric_mics_lmc.py \
    inference.model_path=/path/to/trained_model.pth \
    inference.inference_folder=/path/to/folder/*_test.npy
```

The script will:
- Load the trained model from the specified path
- Find all files matching the inference_folder pattern (supports glob patterns)
- Process all matching .npy files
- Generate and save visualizations for each file
- Save all visualizations to the output directory

### Custom Output Directory

Specify a custom output directory:

```bash
python train_parametric_mics_lmc.py data.input_path=/path/to/image.npy output_dir=/path/to/outputs
```

## Hyperparameter Sweeps

### Single Parameter Sweep

Sweep over different sampling methods:

```bash
python train_parametric_mics_lmc.py --multirun data.input_path=/path/to/image.npy model.sampling=random,coreset,kmeans++,kmeans
```

### Multiple Parameter Sweeps

Sweep over multiple parameters (all combinations):

```bash
python train_parametric_mics_lmc.py --multirun \
    data.input_path=/path/to/image.npy \
    model.sampling=random,coreset \
    model.num_epochs=50,100,200 \
    model.beta=0.0,1.0,5.0
```

### Sweep Over Cluster Configurations

```bash
python train_parametric_mics_lmc.py --multirun \
    data.input_path=/path/to/image.npy \
    model.clusters="[32,64,128]","[64,128,256]","[128,256,512]"
```

## Configuration Override Examples

### Override Model Parameters

```bash
python train_parametric_mics_lmc.py \
    data.input_path=/path/to/image.npy \
    model.num_epochs=200 \
    model.batch_size=2048 \
    model.lr=0.5 \
    model.verbose=true
```

### Enable Clustering with Custom Clusters

```bash
python train_parametric_mics_lmc.py \
    data.input_path=/path/to/image.npy \
    model.cluster=true \
    model.clusters=[32,64,128,256,512] \
    model.beta=1.0 \
    model.temperature=100.0
```

### Disable Visualizations

```bash
python train_parametric_mics_lmc.py \
    data.input_path=/path/to/image.npy \
    save_visualizations=false
```

## Output Structure

The script creates an output directory with the following structure:

```
outputs/
└── parametric_mics_lmc_YYYYMMDD_HHMMSS/
    ├── .hydra/              # Hydra configuration files
    │   ├── config.yaml
    │   └── ...
    ├── config.yaml          # Full configuration
    ├── training_summary.txt # Training summary
    ├── model_<filename>.pth # Trained models
    └── <filename>_viz.png   # Visualizations (if enabled)
```

## Configuration File

The default configuration is in `configs/train_parametric_mics_lmc.yaml`. You can:

1. Edit the default config file directly
2. Override parameters via command line
3. Create custom config files

### Example Custom Config

Create `configs/my_config.yaml`:

```yaml
defaults:
  - train_parametric_mics_lmc

model:
  num_epochs: 200
  batch_size: 2048
  verbose: true

data:
  transpose: true
```

Then use it:

```bash
python train_parametric_mics_lmc.py --config-name=my_config data.input_path=/path/to/image.npy
```

## Advanced Sweep Examples

### Grid Search

```bash
python train_parametric_mics_lmc.py --multirun \
    data.input_path=/path/to/image.npy \
    model.sampling=random,coreset \
    model.num_epochs=50,100 \
    model.beta=0.0,1.0,5.0
```

This will run 2 × 2 × 3 = 12 experiments.

### Sweep with Different Cluster Configurations

```bash
python train_parametric_mics_lmc.py --multirun \
    data.input_path=/path/to/image.npy \
    model.cluster=true \
    model.clusters="[32,64]","[64,128]","[128,256]","[256,512]"
```

## Tips

1. **Use `--multirun` for sweeps**: This enables Hydra's multirun mode for hyperparameter sweeps
2. **Check output directories**: Each run creates a dated output directory
3. **Review config files**: The `.hydra/config.yaml` in each output directory contains the exact configuration used
4. **Enable verbose mode**: Set `model.verbose=true` to see detailed training progress
5. **Continue on error**: Set `continue_on_error=true` to continue processing other files if one fails
6. **Inference mode**: Provide `inference.model_path` and `inference.inference_folder` to run inference instead of training
7. **Training with inference**: Set `inference.inference_folder` during training to automatically run inference after training completes
8. **Glob patterns**: The `inference.inference_folder` parameter supports glob patterns like `/*.npy` or `/*_test.npy`
9. **Model compatibility**: The inference mode automatically loads model configuration from the checkpoint, so you don't need to match all hyperparameters

## Troubleshooting

### Path Issues

- Use absolute paths or paths relative to the script location
- On Windows, use forward slashes or escaped backslashes: `C:/path/to/file.npy` or `C:\\path\\to\\file.npy`

### Memory Issues

- Reduce `model.batch_size` if you run out of memory
- Reduce `model.num_samples` to sample fewer points
- Process files one at a time instead of in a folder

### Hydra Output

- Hydra creates its own output structure in `outputs/` by default
- The script also creates a custom dated directory for easier organization
- Both contain the same information, organized differently

