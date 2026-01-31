#!/usr/bin/env python3
"""
Training and inference script for MSIParametricMiCSLMC with Hydra configuration management.

Supports:
- Training: Single .npy file or folder of .npy files
- Inference: Load trained model and generate visualizations on folder of .npy files
- Hyperparameter sweeps via Hydra
- Output directory with date
- Model saving and visualization
"""

import os
import glob
import numpy as np
from pathlib import Path
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from msi_visual.parametric_mics_lmc import MSIParametricMiCSLMC, create_pcc_model
from msi_visual.quality_metrics import (
    compute_highd_summary,
    save_quality_score_image,
    save_edge_quality_image,
    save_structure_quality_image,
    save_contrast_quality_image,
)


def load_and_preprocess_image(img_path, transpose=False):
    """Load and preprocess a single .npy image file."""
    img = np.load(img_path, mmap_mode=None)
    if img.dtype != np.float32:
        img = img.astype(np.float32, copy=False)
    # In-place sum normalization (no copy); works if img is float (safe here as we just cast to float32 above)
    img_sum = img.sum(axis=-1, keepdims=True)
    np.add(img_sum, 1e-6, out=img_sum)
    np.divide(img, img_sum, out=img)
    if transpose:
        img = img.transpose().transpose(1, 2, 0)[::-1, :, :].copy()
    return img


def compute_uncertainty_map(model, img):
    """Compute a normalized uncertainty (entropy) map from cluster classifiers."""
    if not getattr(model, "visualiation_to_cluster", None):
        raise ValueError("No classifiers available - enable clustering and load classifiers.")

    img_flat = img.reshape(img.shape[0] * img.shape[1], -1)
    mask = img.sum(axis=-1) > 0

    model.model.eval()
    for classifier in model.visualiation_to_cluster:
        classifier.eval()

    batch_size = 1024 * 32
    entropies = []

    with torch.no_grad():
        x_batches = torch.split(torch.from_numpy(img_flat).float(), batch_size)
        for x_batch in x_batches:
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
            features = model.model(x_batch)

            batch_entropy = None
            for classifier in model.visualiation_to_cluster:
                layer = classifier.cuda() if torch.cuda.is_available() else classifier
                logits = layer(features)
                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)
                batch_entropy = entropy if batch_entropy is None else batch_entropy + entropy

            batch_entropy = (batch_entropy / len(model.visualiation_to_cluster)).cpu().numpy()
            entropies.append(batch_entropy)

    entropy_map = np.concatenate(entropies, axis=0).reshape(img.shape[0], img.shape[1])
    entropy_map[~mask] = 0.0

    if mask.any():
        min_val = entropy_map[mask].min()
        max_val = entropy_map[mask].max()
        if max_val > min_val:
            entropy_map = (entropy_map - min_val) / (max_val - min_val)
        else:
            entropy_map = np.zeros_like(entropy_map)
    else:
        entropy_map = np.zeros_like(entropy_map)

    return entropy_map


def save_uncertainty_image(model, img, output_path, filename):
    """Generate and save an uncertainty image (entropy map)."""
    uncertainty = compute_uncertainty_map(model, img)
    uncertainty_uint8 = (uncertainty * 255).clip(0, 255).astype(np.uint8)
    equalized = cv2.equalizeHist(uncertainty_uint8)
    equalized[uncertainty_uint8 == 0] = 0
    img_pil = Image.fromarray(equalized, mode="L")
    img_pil.save(output_path / f"{filename}_uncertainty.png")
    return equalized


def load_trained_model(model_path, cfg):
    """Load a trained model from a saved checkpoint."""
    print(f"\n{'='*80}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*80}\n")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get config from checkpoint or use provided config
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        # Use saved config for model parameters, but allow override from cfg
        model_kwargs = {
            'number_of_points': saved_config.get('model', {}).get('number_of_points', cfg.model.number_of_points),
            'sampling': saved_config.get('model', {}).get('sampling', cfg.model.sampling),
            'num_epochs': saved_config.get('model', {}).get('num_epochs', cfg.model.num_epochs),
            'number_of_components': saved_config.get('model', {}).get('number_of_components', cfg.model.number_of_components),
            'lab_to_rgb': saved_config.get('model', {}).get('lab_to_rgb', cfg.model.lab_to_rgb),
            'cluster': saved_config.get('model', {}).get('cluster', cfg.model.cluster),
            'clusters': saved_config.get('model', {}).get('clusters', cfg.model.clusters if cfg.model.cluster else None),
            'num_samples': saved_config.get('model', {}).get('num_samples', cfg.model.num_samples),
            'beta': saved_config.get('model', {}).get('beta', cfg.model.beta),
            'k_epoch': saved_config.get('model', {}).get('k_epoch', cfg.model.k_epoch),
            'temperature': saved_config.get('model', {}).get('temperature', cfg.model.temperature),
            'lr': saved_config.get('model', {}).get('lr', cfg.model.lr),
            'batch_size': saved_config.get('model', {}).get('batch_size', cfg.model.batch_size),
            'warmup_epochs': saved_config.get('model', {}).get('warmup_epochs', cfg.model.warmup_epochs),
            'factor': saved_config.get('model', {}).get('factor', cfg.model.factor),
            'random_state': saved_config.get('model', {}).get('random_state', cfg.model.random_state),
            'verbose': cfg.model.verbose,
            'cluster_loss_weight': saved_config.get('model', {}).get('cluster_loss_weight', getattr(cfg.model, 'cluster_loss_weight', 1.0)),
            'category_loss_weight': saved_config.get('model', {}).get('category_loss_weight', getattr(cfg.model, 'category_loss_weight', 1.0)),
        }
    else:
        # Fallback to provided config
        model_kwargs = {
            'number_of_points': cfg.model.number_of_points,
            'sampling': cfg.model.sampling,
            'num_epochs': cfg.model.num_epochs,
            'number_of_components': cfg.model.number_of_components,
            'lab_to_rgb': cfg.model.lab_to_rgb,
            'cluster': cfg.model.cluster,
            'clusters': cfg.model.clusters if cfg.model.cluster else None,
            'num_samples': cfg.model.num_samples,
            'beta': cfg.model.beta,
            'k_epoch': cfg.model.k_epoch,
            'temperature': cfg.model.temperature,
            'lr': cfg.model.lr,
            'batch_size': cfg.model.batch_size,
            'warmup_epochs': cfg.model.warmup_epochs,
            'factor': cfg.model.factor,
            'random_state': cfg.model.random_state,
            'verbose': cfg.model.verbose,
            'cluster_loss_weight': getattr(cfg.model, 'cluster_loss_weight', 1.0),
            'category_loss_weight': getattr(cfg.model, 'category_loss_weight', 1.0),
        }

    print(f"Model kwargs: {model_kwargs}")

    # Create model instance
    model = MSIParametricMiCSLMC(**model_kwargs)

    # Ensure clusters are a list of ints when enabled
    if model.cluster and isinstance(model.clusters, str):
        model.clusters = [int(cluster) for cluster in model.clusters.split('-') if cluster]

    # Initialize the model architecture using saved input shape
    input_shape = checkpoint.get('input_shape')
    if input_shape is None:
        raise ValueError("Checkpoint missing input_shape; cannot rebuild model for inference.")
    input_dim = int(input_shape[-1])
    model.model = create_pcc_model(input_dim, model.number_of_components, model.factor)

    # Load model weights
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.eval()

    # Load clustering classifiers if available
    cluster_state_dicts = checkpoint.get('cluster_state_dicts')
    if model.cluster and cluster_state_dicts:
        model.visualiation_to_cluster = []
        for state_dict in cluster_state_dicts:
            out_dim = next(iter(state_dict.values())).shape[0]
            layer = torch.nn.Sequential(torch.nn.Linear(model.number_of_components, out_dim))
            layer.load_state_dict(state_dict)
            if torch.cuda.is_available():
                layer = layer.cuda()
            model.visualiation_to_cluster.append(layer)
    else:
        model.visualiation_to_cluster = []
    
    print(f"Model loaded successfully!")
    if 'input_shape' in checkpoint:
        print(f"Original training input shape: {checkpoint['input_shape']}")
    
    return model


def find_files_with_glob(pattern):
    """Find files matching a glob pattern."""
    if '*' in pattern or '?' in pattern or '[' in pattern:
        # It's a glob pattern
        files = sorted(glob.glob(pattern))
        return [Path(f) for f in files]
    else:
        # It's a directory or file path
        path = Path(pattern)
        if path.is_file():
            return [path]
        elif path.is_dir():
            # Find all .npy files in directory
            files = sorted(glob.glob(str(path / "*.npy")))
            return [Path(f) for f in files]
        else:
            raise ValueError(f"Path does not exist: {pattern}")


def run_inference(cfg, model, input_files, output_dir):
    """Run inference on a list of files using a trained model."""
    print(f"\n{'='*80}")
    print(f"Running inference on {len(input_files)} file(s)")
    print(f"{'='*80}\n")
    
    results = []
    for idx, file_path in enumerate(input_files):
        try:
            print(f"\nProcessing file {idx + 1}/{len(input_files)}: {file_path}")
            
            # Load and preprocess image
            img = load_and_preprocess_image(file_path, transpose=cfg.data.transpose)
            
            # Generate and save visualization
            filename = Path(file_path).stem
            base_viz = model.predict(img)
            save_visualization(model, img, output_dir, filename, viz=base_viz)
            viz_path = output_dir / f"{filename}_viz.png"

            quality_path = None
            edge_quality_path = None
            structure_quality_path = None
            contrast_quality_path = None
            try:
                zero_top_n = int(getattr(cfg.model, "quality_zero_top_n", 10))
                summary_method = getattr(cfg.model, "quality_highd_summary", "pca1")
                window_size = int(getattr(cfg.model, "quality_window_size", 9))
                structure_window = int(getattr(cfg.model, "quality_structure_window", 5))
                summary = compute_highd_summary(img, method=summary_method, random_state=cfg.model.random_state)

                save_quality_score_image(
                    model,
                    img,
                    output_dir,
                    filename,
                    zero_top_n,
                    base_viz=base_viz,
                    show_progress=True,
                )
                quality_path = output_dir / f"{filename}_quality.png"

                save_edge_quality_image(
                    model,
                    img,
                    output_dir,
                    filename,
                    window_size=window_size,
                    highd_summary=summary_method,
                    base_viz=base_viz,
                    summary=summary,
                )
                edge_quality_path = output_dir / f"{filename}_edge_quality.png"

                save_structure_quality_image(
                    model,
                    img,
                    output_dir,
                    filename,
                    window_size=structure_window,
                    highd_summary=summary_method,
                    base_viz=base_viz,
                    summary=summary,
                )
                structure_quality_path = output_dir / f"{filename}_structure_quality.png"

                save_contrast_quality_image(
                    model,
                    img,
                    output_dir,
                    filename,
                    window_size=window_size,
                    highd_summary=summary_method,
                    base_viz=base_viz,
                    summary=summary,
                )
                contrast_quality_path = output_dir / f"{filename}_contrast_quality.png"
            except Exception as e:
                print(f"  ⚠ Quality metrics skipped for {file_path}: {e}")

            results.append({
                "input": str(file_path),
                "viz": str(viz_path),
                "quality": str(quality_path) if quality_path else None,
                "edge_quality": str(edge_quality_path) if edge_quality_path else None,
                "structure_quality": str(structure_quality_path) if structure_quality_path else None,
                "contrast_quality": str(contrast_quality_path) if contrast_quality_path else None,
            })
            
            print(f"  ✓ Visualization saved: {output_dir / f'{filename}_viz.png'}")
            if quality_path:
                print(f"  ✓ Quality saved: {quality_path}")
            if edge_quality_path:
                print(f"  ✓ Edge quality saved: {edge_quality_path}")
            if structure_quality_path:
                print(f"  ✓ Structure quality saved: {structure_quality_path}")
            if contrast_quality_path:
                print(f"  ✓ Contrast quality saved: {contrast_quality_path}")
            
            del img
            
            import gc
            import torch

            # Clean up CUDA memory (if applicable)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Run Python garbage collection
            gc.collect()

        except Exception as e:
            import traceback
            print(f"\n  ✗ Error processing {file_path}: {e}")
            traceback.print_exc()
            if not cfg.continue_on_error:
                raise
            continue
    
    return results

def save_visualization(model, img, output_path, filename, viz=None):
    """Generate and save visualization for an image."""
    if viz is None:
        viz = model.predict(img)
    
    # Apply histogram equalization to each channel
    viz_equalized = np.zeros_like(viz)
    for i in range(viz.shape[2]):
        viz_equalized[:, :, i] = cv2.equalizeHist(viz[:, :, i])
    
    # Save as image
    img_pil = Image.fromarray(viz_equalized)
    img_pil.save(output_path / f"{filename}_viz.png")
    
    return viz_equalized


def train_on_file(cfg, input_path, output_dir, file_idx=0, total_files=1):
    """Train model on a single file."""
    print(f"\n{'='*80}")
    print(f"Training on file {file_idx + 1}/{total_files}: {input_path}")
    print(f"{'='*80}\n")
    
    # Load and preprocess image
    img = load_and_preprocess_image(input_path, transpose=cfg.data.transpose)
    
    # Create model with config parameters
    model_kwargs = {
        'number_of_points': cfg.model.number_of_points,
        'sampling': cfg.model.sampling,
        'num_epochs': cfg.model.num_epochs,
        'number_of_components': cfg.model.number_of_components,
        'lab_to_rgb': cfg.model.lab_to_rgb,
        'cluster': cfg.model.cluster,
        'clusters': [int(cluster) for cluster in cfg.model.clusters.split('-')] if cfg.model.cluster else None,
        'num_samples': cfg.model.num_samples,
        'beta': cfg.model.beta,
        'k_epoch': cfg.model.k_epoch,
        'temperature': cfg.model.temperature,
        'lr': cfg.model.lr,
        'batch_size': cfg.model.batch_size,
        'warmup_epochs': cfg.model.warmup_epochs,
        'factor': cfg.model.factor,
        'random_state': cfg.model.random_state,
        'verbose': cfg.model.verbose,
        'cluster_loss_weight': getattr(cfg.model, 'cluster_loss_weight', 1.0),
        'category_loss_weight': getattr(cfg.model, 'category_loss_weight', 1.0),
    }

    model = MSIParametricMiCSLMC(**model_kwargs)
    
    # Train the model
    model.fit(img)
    
    # Save model
    model_path = output_dir / f"model_{Path(input_path).stem}.pth"
    cluster_state_dicts = None
    if model.cluster and getattr(model, "visualiation_to_cluster", None):
        cluster_state_dicts = [layer.state_dict() for layer in model.visualiation_to_cluster]
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'cluster_state_dicts': cluster_state_dicts,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'input_shape': img.shape,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Generate and save visualization
    if cfg.save_visualizations:
        base_viz = model.predict(img)
        save_visualization(model, img, output_dir, Path(input_path).stem, viz=base_viz)
        print(f"Visualization saved to: {output_dir / f'{Path(input_path).stem}_viz.png'}")
        try:
            zero_top_n = int(getattr(cfg.model, "quality_zero_top_n", 10))
            summary_method = getattr(cfg.model, "quality_highd_summary", "pca1")
            window_size = int(getattr(cfg.model, "quality_window_size", 9))
            structure_window = int(getattr(cfg.model, "quality_structure_window", 5))
            summary = compute_highd_summary(img, method=summary_method, random_state=cfg.model.random_state)

            save_quality_score_image(
                model,
                img,
                output_dir,
                Path(input_path).stem,
                zero_top_n,
                base_viz=base_viz,
                show_progress=True,
            )
            print(f"Quality image saved to: {output_dir / f'{Path(input_path).stem}_quality.png'}")

            save_edge_quality_image(
                model,
                img,
                output_dir,
                Path(input_path).stem,
                window_size=window_size,
                highd_summary=summary_method,
                base_viz=base_viz,
                summary=summary,
            )
            print(f"Edge quality image saved to: {output_dir / f'{Path(input_path).stem}_edge_quality.png'}")

            save_structure_quality_image(
                model,
                img,
                output_dir,
                Path(input_path).stem,
                window_size=structure_window,
                highd_summary=summary_method,
                base_viz=base_viz,
                summary=summary,
            )
            print(f"Structure quality image saved to: {output_dir / f'{Path(input_path).stem}_structure_quality.png'}")

            save_contrast_quality_image(
                model,
                img,
                output_dir,
                Path(input_path).stem,
                window_size=window_size,
                highd_summary=summary_method,
                base_viz=base_viz,
                summary=summary,
            )
            print(f"Contrast quality image saved to: {output_dir / f'{Path(input_path).stem}_contrast_quality.png'}")
        except Exception as e:
            print(f"Warning: Quality metrics skipped for {input_path}: {e}")
    
    return model, model_path


@hydra.main(version_base=None, config_path="configs", config_name="train_parametric_mics_lmc")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    
    # Hydra creates an output directory automatically
    # We'll use it but also create a custom subdirectory with date
    try:
        from hydra.core.hydra_config import HydraConfig
        hydra_output_dir = Path(HydraConfig.get().run.dir)
    except:
        # Fallback if HydraConfig is not available
        hydra_output_dir = Path.cwd() / "outputs"
    
    # Create custom output directory with date
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.output_dir:
        base_output_dir = Path(cfg.output_dir)
    else:
        # Use Hydra's output directory as base
        base_output_dir = hydra_output_dir.parent
    
    output_dir = base_output_dir / f"parametric_mics_lmc_{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Hydra output directory: {hydra_output_dir}")
    print(f"{'='*80}\n")
    
    # Copy Hydra config to our output directory
    hydra_config_path = hydra_output_dir / ".hydra" / "config.yaml"
    if hydra_config_path.exists():
        import shutil
        hydra_dir = output_dir / ".hydra"
        hydra_dir.mkdir(exist_ok=True)
        shutil.copytree(hydra_output_dir / ".hydra", hydra_dir, dirs_exist_ok=True)
        print(f"Hydra configuration copied to: {hydra_dir}")
    
    # Also save config to output directory
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(config=cfg, f=f)
        print(f"Configuration saved to: {config_path}")
    
    # Check if we're in inference mode
    is_inference = (hasattr(cfg, 'inference') and 
                   cfg.inference.model_path and 
                   (cfg.inference.inference_folder or cfg.mode == "inference"))
    
    if is_inference:
        # Inference mode: load model and run inference
        if not hasattr(cfg, 'inference') or not cfg.inference.model_path:
            raise ValueError("Inference mode requires inference.model_path to be specified")
        
        if not cfg.inference.inference_folder:
            raise ValueError("Inference mode requires inference.inference_folder to be specified")
        
        model_path = Path(cfg.inference.model_path)
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")
        
        # Load trained model
        model = load_trained_model(model_path, cfg)
        
        # Find files to process using glob pattern support
        inference_pattern = cfg.inference.inference_folder
        input_files = find_files_with_glob(inference_pattern)
        
        if not input_files:
            raise ValueError(f"No files found matching pattern: {inference_pattern}")
        
        print(f"Found {len(input_files)} file(s) to process for inference")
        
        # Run inference
        visualizations = run_inference(cfg, model, input_files, output_dir)
        
        print(f"\n{'='*80}")
        print(f"Inference completed! Visualizations saved to: {output_dir}")
        print(f"{'='*80}\n")
        
        # Save summary
        summary_path = output_dir / "inference_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Inference Summary\n")
            f.write(f"{'='*80}\n")
            f.write(f"Date: {date_str}\n")
            f.write(f"Model path: {cfg.inference.model_path}\n")
            f.write(f"Inference folder/pattern: {cfg.inference.inference_folder}\n")
            f.write(f"Number of files processed: {len(visualizations)}\n")
            f.write(f"\nConfiguration:\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write(f"\n\nVisualizations saved:\n")
            for item in visualizations:
                f.write(f"  {item['input']} -> {item['viz']}\n")
                if item.get('quality'):
                    f.write(f"    quality -> {item['quality']}\n")
                if item.get('edge_quality'):
                    f.write(f"    edge_quality -> {item['edge_quality']}\n")
                if item.get('structure_quality'):
                    f.write(f"    structure_quality -> {item['structure_quality']}\n")
                if item.get('contrast_quality'):
                    f.write(f"    contrast_quality -> {item['contrast_quality']}\n")
        
        print(f"Summary saved to: {summary_path}")
        
    else:
        # Training mode: determine input files
        if not cfg.data.input_path:
            raise ValueError("Training mode requires data.input_path to be specified")
        
        input_path = Path(cfg.data.input_path)
        
        if input_path.is_file() and input_path.suffix == '.npy':
            # Single file
            input_files = [input_path]
        elif input_path.is_dir():
            # Folder of .npy files
            input_files = sorted([Path(f) for f in glob.glob(str(input_path / "*.npy"))])
            if not input_files:
                raise ValueError(f"No .npy files found in directory: {input_path}")
        else:
            raise ValueError(f"Input path must be a .npy file or directory: {input_path}")
        
        print(f"Found {len(input_files)} file(s) to process for training")
        # Training mode: train models
        # Store only model paths, not the actual models to save memory
        model_paths = []
        for idx, file_path in enumerate(input_files):
            try:
                model, model_path = train_on_file(cfg, file_path, output_dir, idx, len(input_files))
                model_paths.append(model_path)
                # Free the model from memory after saving
                del model
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"\nError processing {file_path}: {e}")
                if not cfg.continue_on_error:
                    raise
                continue
        
        print(f"\n{'='*80}")
        print(f"Training completed! Results saved to: {output_dir}")
        print(f"{'='*80}\n")
        
        # Save summary
        summary_path = output_dir / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"{'='*80}\n")
            f.write(f"Date: {date_str}\n")
            f.write(f"Input path: {cfg.data.input_path}\n")
            f.write(f"Number of files processed: {len(model_paths)}\n")
            f.write(f"\nConfiguration:\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write(f"\n\nModels saved:\n")
            for model_path in model_paths:
                f.write(f"  - {model_path}\n")
        
        print(f"Summary saved to: {summary_path}")
        
        # Run inference after training if inference_folder is specified
        if hasattr(cfg, 'inference') and cfg.inference.inference_folder and model_paths:
            print(f"\n{'='*80}")
            print(f"Running inference after training...")
            print(f"{'='*80}\n")
            
            # Use the last trained model - load it from disk
            trained_model_path = model_paths[-1]
            print(f"Loading model from: {trained_model_path}")
            trained_model = load_trained_model(trained_model_path, cfg)
            
            # Find files to process using glob pattern support
            inference_pattern = cfg.inference.inference_folder
            inference_files = find_files_with_glob(inference_pattern)
            
            if not inference_files:
                print(f"Warning: No files found matching pattern: {inference_pattern}")
            else:
                print(f"Found {len(inference_files)} file(s) to process for inference")
                
                # Run inference using the trained model
                visualizations = run_inference(cfg, trained_model, inference_files, output_dir)
                
                # Free the model after inference
                del trained_model
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"\n{'='*80}")
                print(f"Inference after training completed! Visualizations saved to: {output_dir}")
                print(f"{'='*80}\n")
                
                # Update summary with inference results
                inference_summary_path = output_dir / "inference_after_training_summary.txt"
                with open(inference_summary_path, 'w') as f:
                    f.write(f"Inference After Training Summary\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Date: {date_str}\n")
                    f.write(f"Model used: {trained_model_path}\n")
                    f.write(f"Inference folder/pattern: {cfg.inference.inference_folder}\n")
                    f.write(f"Number of files processed: {len(visualizations)}\n")
                    f.write(f"\nConfiguration:\n")
                    f.write(OmegaConf.to_yaml(cfg))
                    f.write(f"\n\nVisualizations saved:\n")
                    for item in visualizations:
                        f.write(f"  {item['input']} -> {item['viz']}\n")
                        if item.get('quality'):
                            f.write(f"    quality -> {item['quality']}\n")
                        if item.get('edge_quality'):
                            f.write(f"    edge_quality -> {item['edge_quality']}\n")
                        if item.get('structure_quality'):
                            f.write(f"    structure_quality -> {item['structure_quality']}\n")
                        if item.get('contrast_quality'):
                            f.write(f"    contrast_quality -> {item['contrast_quality']}\n")
                
                print(f"Inference summary saved to: {inference_summary_path}")


if __name__ == "__main__":
    main()

