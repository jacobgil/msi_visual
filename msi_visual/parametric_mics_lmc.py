import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
import cv2
import tqdm
from typing import Optional, List, Any
import random
import time
from sklearn.decomposition import PCA
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from sklearn.cluster import KMeans, kmeans_plusplus
from msi_visual.utils import normalize


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            torch.nn.GELU()
        )

    def forward(self, x):
        return x + self.layer(x)


def create_pcc_model(input_dim, n_components=3, factor=1.0):
    """
    Create a PCC model with the architecture from supervised_dr.ipynb.
    
    Args:
        input_dim: Input dimension (number of features)
        n_components: Output dimension (number of components)
        factor: Scaling factor for model size (default 1.0)
        
    Returns:
        torch.nn.Module: The model
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, int(2048 * factor)),
        torch.nn.BatchNorm1d(int(2048 * factor)),
        torch.nn.GELU(),
        ResidualBlock(int(2048 * factor)),
        torch.nn.BatchNorm1d(int(2048 * factor)),
        torch.nn.GELU(),
        ResidualBlock(int(2048 * factor)),
        torch.nn.Linear(int(2048 * factor), n_components)
    )
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def correlation(
    pred: torch.Tensor, target: torch.Tensor, dim: Optional[int] = None
) -> torch.Tensor:
    """
    Compute correlation between two tensors.

    Args:
        pred: Prediction tensor
        target: Target tensor
        dim: Dimension along which to compute correlation. If None, treats tensors as 1D.

    Returns:
        Correlation coefficient as a tensor
    """
    if dim is None:
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return (pred * target).sum()
    else:
        pred = pred - pred.mean(dim=dim)[:, None]
        pred = pred / pred.norm(dim=dim)[:, None]
        target = target - target.mean(dim=dim)[:, None]
        target = target / target.norm(dim=dim)[:, None]
        return (pred * target).sum(dim=dim).mean()


def random_pixel_sampling(X_raw, approx_budget, height, width, roi_mask=None, rng_seed=42):
    """
    Fast path: sample random valid pixel indices (no PCA/SLIC). Use when superpixel is too slow.
    Returns (sampled_coords, sampled_flat_indices, None) for API compatibility.
    """
    mask = X_raw.max(axis=-1) > 0
    valid_flat = np.where(mask.ravel())[0]
    if roi_mask is not None:
        roi_flat = np.asarray(roi_mask, dtype=bool).reshape(height, width).ravel()
        valid_flat = valid_flat[roi_flat[valid_flat]]
    n = len(valid_flat)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64), None
    rng = np.random.default_rng(rng_seed)
    k = min(approx_budget, n)
    chosen_flat = rng.choice(valid_flat, size=k, replace=False)
    rows, cols = np.unravel_index(chosen_flat, (height, width))
    sampled_coords = np.column_stack((rows, cols))
    return sampled_coords, chosen_flat, None


def uniform_superpixel_sampling(X, X_raw, approx_budget=5000, 
                               init_superpixels=256, 
                               init_points_per_superpixel=8, 
                               compactness=10, 
                               max_iters=8, 
                               verbose=False, 
                               visualize=False,
                               rng_seed=42,
                               roi_mask=None,
                               pca_fit_step=4):
    """
    Uniformly sample points from superpixels with a target total sample budget.
    Will adaptively increase superpixels or points per superpixel if not enough points.
    
    Args:
        X: (H, W, C) array, main image stack
        X_raw: (H*W, C), flattened image stack
        approx_budget: int, target number of sampled points
        init_superpixels: initial n_superpixels to start from
        init_points_per_superpixel: initial samples per superpixel
        compactness: passed to slic
        max_iters: number of tries before giving up
        verbose: print search log
        visualize: show matplotlib overlay
        rng_seed: for random generator
        roi_mask: optional (H, W) boolean array; if provided, sampling is restricted to ROI.
        pca_fit_step: int, subsample step for PCA fit (X_raw[::pca_fit_step, :]); 1 = use all pixels.

    Returns:
        sampled_coords: (N, 2) array of sampled coordinates (row, col)
        sampled_flat_indices: (N, ) array for X_raw selection
        mask_labels: SLIC label map
    """
    mask = X.sum(axis=-1) > 0
    height, width = X.shape[:2]
    pca_fit_step = max(1, int(pca_fit_step))

    # 1. PCA for RGB mapping for visualization and SLIC basis
    t0 = time.perf_counter()
    pca = PCA(n_components=3)
    pca.fit(X_raw[::pca_fit_step, :])
    X_pca = pca.transform(X_raw)
    X_rgb = X_pca - X_pca.min(axis=0)
    X_rgb = X_rgb / (X_rgb.max(axis=0) + 1e-8)
    X_rgb = (X_rgb * 255).clip(0, 255).astype(np.uint8)
    X_rgb_img = X_rgb.reshape(height, width, 3)
    X_rgb_float = img_as_float(X_rgb_img)
    t_pca = time.perf_counter() - t0
    print(f"[Superpixel] PCA (fit step={pca_fit_step}): {t_pca:.2f}s")

    mask_img = mask.reshape(height, width)
    valid_pixel_bool = mask_img.astype(bool)
    if roi_mask is not None:
        valid_pixel_bool = valid_pixel_bool & np.asarray(roi_mask, dtype=bool).reshape(height, width)
    sampled_coords = []

    n_superpixels = init_superpixels
    points_per_superpixel = init_points_per_superpixel

    # Adaptive search for enough points
    for attempt in range(max_iters):
        if verbose:
            print(f"[Superpixel Select] Attempt {attempt+1}, n_segments={n_superpixels}, pts_per_superpixel={points_per_superpixel}")
        
        # 2. Superpixel segmentation
        t0 = time.perf_counter()
        mask_labels = slic(X_rgb_float, n_segments=n_superpixels, compactness=compactness, channel_axis=2, start_label=0)
        t_slic = time.perf_counter() - t0
        print(f"[Superpixel] SLIC (n_segments={n_superpixels}): {t_slic:.2f}s")
        unique_labels = np.unique(mask_labels)
        # Vectorized: one pass over valid pixels and their labels (avoids N_superpixels full-image argwhere)
        valid_flat = np.where(valid_pixel_bool.ravel())[0]
        labels_valid = mask_labels.ravel()[valid_flat]
        rows, cols = np.unravel_index(valid_flat, (height, width))
        result_coords = []
        rng = np.random.default_rng(rng_seed)
        for lbl in unique_labels:
            idx = labels_valid == lbl
            n = idx.sum()
            if n == 0:
                continue
            coords_valid = np.column_stack((rows[idx], cols[idx]))
            if n <= points_per_superpixel:
                chosen = coords_valid
            else:
                idx_sort = np.lexsort((coords_valid[:, 1], coords_valid[:, 0]))
                coords_valid_sorted = coords_valid[idx_sort]
                steps = max(1, n // points_per_superpixel)
                chosen = coords_valid_sorted[::steps][:points_per_superpixel]
            result_coords.append(chosen)
            
        if len(result_coords) == 0:
            sampled_coords = np.empty((0, 2), dtype=int)
        else:
            sampled_coords = np.concatenate(result_coords, axis=0)
        
        if verbose:
            print(f"    Collected {sampled_coords.shape[0]} points (budget={approx_budget})")
        
        # If meets budget, return
        if sampled_coords.shape[0] >= approx_budget:
            break
        # else, adaptively increase superpixels or points per superpixel
        if attempt % 2 == 0:
            # alternate between increasing points per superpixel
            points_per_superpixel = int(points_per_superpixel * 1.5)
        else:
            n_superpixels = int(n_superpixels * 1.5) + 8  # +8 to ensure new superpixels appear

    # Final to flat indices for X_raw
    if sampled_coords.shape[0] == 0:
        sampled_flat_indices = np.empty((0,), dtype=int)
    else:
        sampled_flat_indices = np.ravel_multi_index((sampled_coords[:, 0], sampled_coords[:, 1]), (height, width))

    if verbose:
        print("Returned", sampled_flat_indices.shape)

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 10))
        img_with_boundaries = mark_boundaries(X_rgb_img, mask_labels, color=(1, 0, 0), mode='thick')
        ax.imshow(img_with_boundaries)
        if len(sampled_coords) > 0:
            ax.scatter(sampled_coords[:, 1], sampled_coords[:, 0], s=8, c='yellow', alpha=0.7, label='Superpixel Samples')
        ax.axis('off')
        ax.set_title(f"Superpixel Overlay: Boundaries, Superpixel Uniform Samples (yellow);\nTotal points: {sampled_coords.shape[0]}")
        ax.legend()
        plt.show()


    return sampled_coords, sampled_flat_indices, mask_labels


def _build_category_labels(
    height: int,
    width: int,
    sampled_flat_indices: np.ndarray,
    category_annotations: List[dict],
) -> np.ndarray:
    """
    Assign a category to each sampled point from polygon annotations.
    category_annotations: list of {"polygon": [[x, y], ...], "category": int}
    with (x, y) = (col, row). Points inside no polygon get -1 (ignore).
    First matching polygon wins.
    """
    n_sampled = len(sampled_flat_indices)
    out = np.full(n_sampled, -1, dtype=np.int64)
    if not category_annotations:
        return out
    # Build one mask per annotation (order matters: first match wins)
    for ann in category_annotations:
        poly = ann.get("polygon") or ann.get("points")
        cat = ann.get("category", 0)
        if not poly or len(poly) < 3:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array(poly, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        mask_bool = mask.astype(bool)
        rows = sampled_flat_indices // width
        cols = sampled_flat_indices % width
        inside = mask_bool[rows, cols]
        # Only assign where still unlabeled
        out[(out == -1) & inside] = cat
    return out


class MSIParametricMiCSLMC:
    def __init__(
            self,
            model: torch.nn.Module = None,
            number_of_points=500,
            sampling="coreset",
            num_epochs=200,
            number_of_components=3,
            lab_to_rgb=True,
            cluster=False,
            clusters=None,
            num_samples=5000,
            beta=5.0,
            k_epoch=1,
            temperature=10.0,
            lr=1.0,
            batch_size=4096 * 2,
            warmup_epochs=10,
            factor=1.0,
            random_state=42,
            verbose=False,
            cluster_loss_weight=1.0,
            category_loss_weight=1.0,
            pixel_sampling="superpixel",
            pca_fit_step=4,
            cluster_on_pca=False,
            cluster_pca_dims=50):
        self.model = model
        self.verbose = verbose
        self.factor = factor
        self.num_epochs = num_epochs
        self.sampling = sampling
        self.number_of_points = number_of_points
        self.number_of_components = number_of_components
        self.lab_to_rgb = lab_to_rgb
        self.cluster = cluster
        self.clusters = clusters if clusters is not None else []
        self.num_samples = num_samples
        self.beta = beta
        self.k_epoch = k_epoch
        self.temperature = temperature
        self.lr = lr
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.random_state = random_state
        self.step_counter = 0
        self.scheduler = None
        self.cluster_loss_weight = float(cluster_loss_weight)
        self.category_loss_weight = float(category_loss_weight)
        self.pixel_sampling = str(pixel_sampling).lower() if pixel_sampling else "superpixel"
        self.pca_fit_step = max(1, int(pca_fit_step))
        self.cluster_on_pca = bool(cluster_on_pca)
        self.cluster_pca_dims = max(1, int(cluster_pca_dims))
        
        # Seed everything for reproducibility
        self.seed_everything()

    def seed_everything(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __repr__(self):
        return f"Parametric PCC Optimization: num_epochs: {self.num_epochs} \
            sampling: {self.sampling} number_of_points:{self.number_of_points} cluster: {self.cluster}"

    def release_resources(self):
        """Release GPU and large tensors so the object can be collected and memory freed. Call before dropping the model reference."""
        try:
            if self.model is not None:
                self.model.cpu()
            if getattr(self, "reference_points", None) is not None:
                del self.reference_points
                self.reference_points = None
            if getattr(self, "euclidean", None) is not None:
                del self.euclidean
                self.euclidean = None
            if getattr(self, "mask", None) is not None:
                del self.mask
                self.mask = None
            if getattr(self, "visualiation_to_cluster", None):
                for layer in self.visualiation_to_cluster:
                    try:
                        layer.cpu()
                    except Exception:
                        pass
                self.visualiation_to_cluster = []
            if getattr(self, "category_head", None) is not None:
                try:
                    self.category_head.cpu()
                except Exception:
                    pass
                self.category_head = None
            if getattr(self, "category_labels", None) is not None:
                self.category_labels = None
            if getattr(self, "optim", None) is not None:
                self.optim = None
            if getattr(self, "scheduler", None) is not None:
                self.scheduler = None
            # Drop large numpy arrays so they can be collected
            if getattr(self, "sampled_data", None) is not None:
                self.sampled_data = None
            if getattr(self, "sampled_flat_indices", None) is not None:
                self.sampled_flat_indices = None
        except Exception:
            pass

    def get_reference_points(self, data, Np):
        """Reduces (NxD) data matrix from N to Np data points.

        Args:
            data: ndarray of shape [N, D]
            Np: number of data points in the coreset
        Returns:
            coreset: ndarray of shape [Np, D]
            weights: 1darray of shape [Np, 1]
        """
        N = data.shape[0]
        method = self.sampling
        rng = np.random.RandomState(self.random_state)

        if method == "random":
            return rng.choice(N, Np, replace=False)

        elif method == "kmeans++":
            _, indices = kmeans_plusplus(data, n_clusters=Np, random_state=self.random_state)
            return indices

        elif method == "kmeans":
            kmeans = KMeans(n_clusters=Np, random_state=self.random_state, n_init="auto").fit(data)
            return pairwise_distances(data, kmeans.cluster_centers_).argmin(axis=0)

        elif method == "coreset":
            # compute mean
            u = np.mean(data, axis=0)
            # compute proposal distribution
            q = np.linalg.norm(data - u, axis=1)**2
            q_sum = np.sum(q)
            d = q / q_sum
            q = 0.5 * (d + 1.0 / N)
            # get sample and fill coreset
            return rng.choice(N, Np, p=q)

    def set_image(self, img, roi_mask=None, category_annotations=None):
        if self.verbose:
            print(f"[set_image] Starting image setup: shape={img.shape}, roi_mask={roi_mask is not None}, category_annotations={bool(category_annotations)}")
        self.img = img
        self.reshaped = self.img.reshape(
            self.img.shape[0] * self.img.shape[1], -1)
        self.img_mask = self.img.max(axis=-1) > 0
        if self.verbose:
            print(f"[set_image] Reshaped image: {self.reshaped.shape}")
        
        # Pixel sampling: "superpixel" (slower, spread over segments) or "random" (fast)
        H, W = self.img.shape[0], self.img.shape[1]
        if self.pixel_sampling == "random":
            _, sampled_flat_indices, _ = random_pixel_sampling(
                self.reshaped, self.num_samples, H, W,
                roi_mask=roi_mask, rng_seed=self.random_state
            )
        else:
            _, sampled_flat_indices, _ = uniform_superpixel_sampling(
                self.img,
                self.reshaped,
                approx_budget=self.num_samples,
                init_superpixels=512,
                init_points_per_superpixel=8,
                compactness=10,
                max_iters=8,
                verbose=self.verbose,
                visualize=self.verbose,
                rng_seed=self.random_state,
                roi_mask=roi_mask,
                pca_fit_step=getattr(self, "pca_fit_step", 4),
            )
        self.sampled_data = self.reshaped[sampled_flat_indices]
        self.sampled_flat_indices = sampled_flat_indices
        self.roi_mask = roi_mask  # store for ROI-based normalization in predict
        if self.verbose:
            print(f"[set_image] Superpixel sampling completed: sampled {self.sampled_data.shape[0]} points from {self.reshaped.shape[0]} total")
        
        # Initialize model if not provided
        if self.model is None:
            input_dim = self.reshaped.shape[1]
            self.model = create_pcc_model(input_dim, self.number_of_components, self.factor)
            if self.verbose:
                print(f"[set_image] Model initialized: input_dim={input_dim}, n_components={self.number_of_components}, factor={self.factor}")
        else:
            if self.verbose:
                print(f"[set_image] Using provided model")
        
        # Resample operates on the sampled_data, not the entire image
        # Validate that number_of_points doesn't exceed sampled data size
        max_points = min(self.number_of_points, self.sampled_data.shape[0])
        if max_points < self.number_of_points:
            if self.verbose:
                print(f"Warning: number_of_points ({self.number_of_points}) exceeds sampled data size ({self.sampled_data.shape[0]}). Using {max_points} points.")
        self.resample(number_of_points=max_points)
        if self.verbose:
            print(f"[set_image] Reference points resampled: {max_points} points using '{self.sampling}' method")
        
        self.mask_np = self.reshaped.max(axis=-1) > 0
        self.mask = torch.from_numpy(self.mask_np).float()
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

        # Category annotations: assign labels from polygon masks (supervised)
        height, width = self.img.shape[0], self.img.shape[1]
        self.category_labels = None
        self.category_head = None
        if category_annotations and len(category_annotations) > 0:
            self.category_labels = _build_category_labels(
                height, width, self.sampled_flat_indices, category_annotations
            )
            num_categories = max(ann.get("category", 0) for ann in category_annotations) + 1
            self.category_head = torch.nn.Sequential(
                torch.nn.Linear(self.number_of_components, num_categories)
            )
            if torch.cuda.is_available():
                self.category_head = self.category_head.cuda()
            if self.verbose:
                n_labeled = np.sum(self.category_labels >= 0)
                print(f"[set_image] Category annotations: {len(category_annotations)} polygons, {num_categories} classes, {n_labeled}/{len(self.category_labels)} points labeled")
        else:
            self.category_labels = None
            self.category_head = None

        # Initialize clustering classifiers if enabled
        if self.cluster and len(self.clusters) > 0:
            if self.verbose:
                print(f"[set_image] Initializing clustering with {len(self.clusters)} cluster levels: {self.clusters}")
            self.visualiation_to_cluster = []
            for k in self.clusters:
                layer = torch.nn.Sequential(
                    torch.nn.Linear(self.number_of_components, k)
                )
                if torch.cuda.is_available():
                    layer = layer.cuda()
                self.visualiation_to_cluster.append(layer)
            
            # Create cluster labels: KMeans on raw data or on PCA-reduced data
            data_for_cluster = self.sampled_data
            if getattr(self, "cluster_on_pca", False):
                n_comp = min(
                    self.cluster_pca_dims,
                    self.sampled_data.shape[1],
                    max(1, self.sampled_data.shape[0] - 1),
                )
                pca_cluster = PCA(n_components=n_comp, random_state=self.random_state)
                data_for_cluster = pca_cluster.fit_transform(self.sampled_data)
                if self.verbose:
                    print(f"[set_image] Clustering on PCA-reduced data: {self.sampled_data.shape[1]} -> {n_comp} dims")
            self.cluster_labels = []
            for k in self.clusters:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                labels = kmeans.fit_predict(data_for_cluster)
                self.cluster_labels.append(labels)
            if self.verbose:
                print(f"[set_image] Clustering initialization completed")
        else:
            self.visualiation_to_cluster = []
            self.cluster_labels = []
        
        # Create optimizer with model and classifier parameters
        # Following the pattern from parametric_pcc.py
        params = [{"params": self.model.parameters(), "weight_decay": 0}]
        if self.cluster and len(self.visualiation_to_cluster) > 0:
            for layer in self.visualiation_to_cluster:
                params.append({"params": layer.parameters(), "weight_decay": 0})
        if self.category_head is not None:
            params.append({"params": self.category_head.parameters(), "weight_decay": 0})
        
        self.optim = torch.optim.AdamW(params, lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, lr_lambda=self._lr_lambda
        )
        if self.verbose:
            print(f"[set_image] Optimizer and scheduler initialized: lr={self.lr}, warmup_epochs={self.warmup_epochs}")
            print(f"[set_image] Image setup completed successfully")
    
    def update_roi_sampling(self, roi_mask):
        """
        Update sampling to ROI only: resample points, re-fit cluster labels, re-initialize cluster heads.
        The embedding model is kept (not re-initialized) so full-image structure is preserved;
        only the cluster heads are re-initialized to learn the new ROI cluster assignments.
        Optimizer is recreated so it tracks the new head parameters.
        """
        if self.img is None or self.reshaped is None:
            raise ValueError("Call set_image() first before update_roi_sampling()")
        if self.verbose:
            print(f"[update_roi_sampling] Updating sampling to ROI (keep model, resample + re-init heads)")
        H, W = self.img.shape[0], self.img.shape[1]
        if getattr(self, "pixel_sampling", "superpixel") == "random":
            _, sampled_flat_indices, _ = random_pixel_sampling(
                self.reshaped, self.num_samples, H, W,
                roi_mask=roi_mask, rng_seed=self.random_state
            )
        else:
            _, sampled_flat_indices, _ = uniform_superpixel_sampling(
                self.img,
                self.reshaped,
                approx_budget=self.num_samples,
                init_superpixels=512,
                init_points_per_superpixel=8,
                compactness=10,
                max_iters=8,
                verbose=self.verbose,
                visualize=False,
                rng_seed=self.random_state,
                roi_mask=roi_mask,
                pca_fit_step=getattr(self, "pca_fit_step", 4),
            )
        self.sampled_data = self.reshaped[sampled_flat_indices]
        self.sampled_flat_indices = sampled_flat_indices
        max_points = min(self.number_of_points, self.sampled_data.shape[0])
        self.indices = None  # force resample to recompute from new sampled_data
        self.resample(number_of_points=max_points)
        if self.cluster and len(self.clusters) > 0:
            data_for_cluster = self.sampled_data
            if getattr(self, "cluster_on_pca", False):
                n_comp = min(
                    self.cluster_pca_dims,
                    self.sampled_data.shape[1],
                    max(1, self.sampled_data.shape[0] - 1),
                )
                pca_cluster = PCA(n_components=n_comp, random_state=self.random_state)
                data_for_cluster = pca_cluster.fit_transform(self.sampled_data)
            self.cluster_labels = []
            for k in self.clusters:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state)
                labels = kmeans.fit_predict(data_for_cluster)
                self.cluster_labels.append(labels)
            # Re-initialize cluster heads so they learn the new ROI cluster structure
            self.visualiation_to_cluster = []
            for k in self.clusters:
                layer = torch.nn.Sequential(
                    torch.nn.Linear(self.number_of_components, k)
                )
                if torch.cuda.is_available():
                    layer = layer.cuda()
                self.visualiation_to_cluster.append(layer)
            # Recreate optimizer with same model params + new head params
            params = [{"params": self.model.parameters(), "weight_decay": 0}]
            for layer in self.visualiation_to_cluster:
                params.append({"params": layer.parameters(), "weight_decay": 0})
            self.optim = torch.optim.AdamW(params, lr=self.lr)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optim, lr_lambda=self._lr_lambda
            )
            if self.verbose:
                print(f"[update_roi_sampling] Cluster labels and heads re-initialized for {len(self.clusters)} levels")
    
    def _lr_lambda(self, epoch: int) -> float:
        """ Learning rate scheduler with warmup """
        if epoch < self.warmup_epochs:
            return float(epoch + 1) / float(self.warmup_epochs)
        return 1.0

    def fit(self, X):
        """
        Fit the model on image data X.
        
        This method combines set_image() and training in one call, following
        the sklearn-like API pattern.
        
        Args:
            X: numpy array of shape (H, W, C) - the image data to train on
            verbose: if True, print progress updates during training
            
        Returns:
            self: Returns self for method chaining
        """
        # Initialize the model with the image
        self.set_image(X)
        if self.verbose:
            print(f"[fit] Starting training for {self.num_epochs} epochs...")
        
        # Train the model
        if self.verbose:
            print(f"Training ParametricPCC for {self.num_epochs} epochs...")
        
        for epoch in tqdm.tqdm(range(self.num_epochs)):
            self.step()
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs} completed")
        
        if self.verbose:
            print("Training completed!")
        if self.verbose:
            print(f"[fit] Training completed after {self.num_epochs} epochs")
        
        return self

    def resample(self, number_of_points):
        # Following the pattern from parametric_pcc.py
        # Operate on sampled_data, not the entire image
        if not hasattr(self, 'indices') or self.indices is None:
            sampled_indices = self.get_reference_points(
                self.sampled_data, number_of_points)
            self.indices = sampled_indices
            if self.verbose:
                print(f"[resample] Selected {len(sampled_indices)} reference points using '{self.sampling}' method")

        reference_points = self.sampled_data[self.indices, :]
        self.reference_points = torch.from_numpy(reference_points)
        if torch.cuda.is_available():
            self.reference_points = self.reference_points.cuda()
        if self.verbose:
            print(f"[resample] Reference points shape: {self.reference_points.shape}")

        # Use chebyshev/euclidean computation from parametric_pcc.py
        # Compute distances from sampled data to reference points
        if self.verbose:
            print(f"[resample] Computing distance matrices for {self.sampled_data.shape[0]} points...")
        chebyshev = pairwise_distances(
            self.sampled_data, reference_points, metric='chebyshev')
        chebyshev = chebyshev.argsort().argsort()

        euclidean = pairwise_distances(
            self.sampled_data, reference_points, metric='cosine')
        euclidean = euclidean.argsort().argsort()
        euclidean = (euclidean + chebyshev) / 2
        
        self.euclidean = torch.from_numpy(euclidean)
        if torch.cuda.is_available():
            self.euclidean = self.euclidean.cuda()
        if self.verbose:
            print(f"[resample] Distance matrices computed: shape={self.euclidean.shape}")

    def step(self):
        """
        Compute one optimization epoch using mini-batches.
        Following the pattern from parametric_pcc.py
        """
        batch_size = self.batch_size
        n_samples = self.sampled_data.shape[0]

        # Shuffle indices for random batching
        batch_indices = torch.randperm(n_samples)

        num_batches = (n_samples + batch_size - 1) // batch_size
        if self.step_counter == 0 and self.verbose:
            print(f"[step] Starting optimization: {num_batches} batches, batch_size={batch_size}, n_samples={n_samples}")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_mask = batch_indices[start_idx:end_idx]

            x = torch.from_numpy(self.sampled_data[batch_mask])
            if torch.cuda.is_available():
                x = x.cuda()
            
            batch_outputs = self.model(x)
            batch_reference_points = self.model(self.reference_points)

            loss = 0

            # Get reference points for this batch
            low_d_distances = torch.cdist(batch_outputs, batch_reference_points)

            high_d_distances = self.euclidean[batch_mask]

            # Clustering loss (weighted)
            if self.cluster and len(self.visualiation_to_cluster) > 0:
                cluster_loss_sum = 0.0
                for i in range(len(self.visualiation_to_cluster)):
                    layer = self.visualiation_to_cluster[i]
                    clusters = self.cluster_labels[i]
                    batch_clusters = torch.from_numpy(clusters[batch_mask]).long()
                    if torch.cuda.is_available():
                        batch_clusters = batch_clusters.cuda()

                    layer_output = layer(batch_outputs)
                    cluster_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)(
                        layer_output / self.temperature, batch_clusters
                    )
                    cluster_loss_sum = cluster_loss_sum + cluster_loss
                loss = loss + self.cluster_loss_weight * (cluster_loss_sum / len(self.visualiation_to_cluster))

            # Category annotation loss / supervised loss (weighted)
            if self.category_head is not None and self.category_labels is not None:
                batch_cats = torch.from_numpy(self.category_labels[batch_mask]).long()
                if torch.cuda.is_available():
                    batch_cats = batch_cats.cuda()
                cat_output = self.category_head(batch_outputs)
                cat_loss = torch.nn.functional.cross_entropy(
                    cat_output / self.temperature, batch_cats, ignore_index=-1
                )
                loss = loss + self.category_loss_weight * cat_loss

            # Correlation loss (computed every k_epoch steps)
            if self.step_counter % self.k_epoch == self.k_epoch - 1:
                correlation_loss = -correlation(low_d_distances, high_d_distances).mean()

                if correlation_loss > 0 and self.verbose:
                    print(f"Correlation loss is positive: {correlation_loss}")

                alpha = -float(correlation_loss.detach().cpu().numpy())
                alpha = max(alpha, 1e-6)

                if self.cluster and len(self.visualiation_to_cluster) > 0:
                    loss = loss + correlation_loss * self.beta / alpha
                elif self.category_head is not None:
                    loss = loss + correlation_loss * self.beta / alpha
                else:
                    loss = correlation_loss

            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optim.step()

        # Update scheduler once per epoch (not per batch)
        # if self.scheduler is not None:
        #     self.scheduler.step()

        # Update k_epoch similar to parametric_pcc.py
        if self.step_counter % self.k_epoch == self.k_epoch - 1:
            # Linearly reduce k_epoch to 1 after 90% of epochs
            old_k_epoch = self.k_epoch
            self.k_epoch = max(1, int(self.k_epoch * 0.92))
            if old_k_epoch != self.k_epoch and self.verbose:
                print(f"[step] Updated k_epoch: {old_k_epoch} -> {self.k_epoch}")

        self.step_counter += 1

    def predict(self, img):
        """
        Predict visualization for an image using the trained model.
        Following the pattern from visualize function in supervised_dr.ipynb
        """
        if self.model is None:
            raise ValueError("Model must be initialized. Call set_image() first or provide a model in __init__.")
        
        # Set up image data without redoing clustering
        if self.verbose:
            print(f"[predict] Starting prediction for image shape: {img.shape}")
        self.img = img
        reshaped = self.img.reshape(
            self.img.shape[0] * self.img.shape[1], -1)
        if self.verbose:
            print(f"[predict] Reshaped to: {reshaped.shape}")
        
        # Generate visualization
        self.model.eval()
        with torch.no_grad():
            batch_size = 1024 * 32
            results = []
            x_batches = torch.split(torch.from_numpy(reshaped).float(), batch_size)
            for x_batch in x_batches:
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                batch_result = self.model(x_batch).cpu()
                results.append(batch_result)
            result = torch.cat(results).reshape((self.img.shape[0], self.img.shape[1], self.number_of_components)).numpy()
        if self.verbose:
            print(f"[predict] Model inference completed: result shape={result.shape}")

        mask = self.img.sum(axis=-1) > 0
        # Use ROI-based normalization when model was trained on ROI only, so color scale reflects ROI stats
        norm_mask = None
        if getattr(self, "roi_mask", None) is not None:
            norm_mask = mask & self.roi_mask
        global_contrast = np.uint8(255 * normalize(result, mask=norm_mask))
        global_contrast[mask == 0] = 0
        
        if self.lab_to_rgb and self.number_of_components == 3:
            global_contrast = cv2.cvtColor(global_contrast, cv2.COLOR_LAB2RGB)
            if self.verbose:
                print(f"[predict] Converted LAB to RGB")
        global_contrast[mask == 0] = 0
        if self.verbose:
            print(f"[predict] Prediction completed: output shape={global_contrast.shape}")
        
        return global_contrast

    

    def uncertainty2(self, img):
        """
        Computes the average entropy per pixel across all classifiers.
        Returns an entropy map of shape (H, W).
        """
        if self.model is None:
            raise ValueError("Model must be initialized. Call set_image() first or provide a model in __init__.")
        if not hasattr(self, "visualiation_to_cluster") or len(self.visualiation_to_cluster) == 0:
            raise ValueError("No classifiers available - enable clustering and call set_image() first.")

        if self.verbose:
            print(f"[uncertainty] Starting uncertainty prediction for image shape: {img.shape}")
        self.img = img
        reshaped = self.img.reshape(
            self.img.shape[0] * self.img.shape[1], -1)
        if self.verbose:
            print(f"[uncertainty] Reshaped to: {reshaped.shape}")

        # Get visualizations for all pixels
        self.model.eval()
        with torch.no_grad():
            batch_size = 1024 * 32
            results = []
            x_batches = torch.split(torch.from_numpy(reshaped).float(), batch_size)
            for x_batch in x_batches:
                if torch.cuda.is_available():
                    x_batch = x_batch.cuda()
                batch_result = self.model(x_batch)
                if torch.cuda.is_available():
                    batch_result = batch_result.cpu()
                results.append(batch_result)
            y = torch.cat(results).reshape((reshaped.shape[0], -1))  # (num_pixels, n_components)

        # Compute average entropy per pixel across all classifiers
        # Each classifier gives logits for different k clusters
        entropies = []
        for idx, classifier in enumerate(self.visualiation_to_cluster):
            classifier.eval()
            with torch.no_grad():
                batch_outs = []
                x_batches = torch.split(y, batch_size)
                for x_batch in x_batches:
                    if torch.cuda.is_available():
                        x_batch = x_batch.cuda()
                        classifier_ = classifier.cuda()
                    else:
                        classifier_ = classifier
                    logits = classifier_(x_batch)
                    logits = logits.cpu().numpy()
                    # Turn logits into probabilities with softmax
                    # Avoid overflow: subtract max for numerical stability
                    logits = logits - np.max(logits, axis=1, keepdims=True)
                    exps = np.exp(logits)
                    probs = exps / np.sum(exps, axis=1, keepdims=True)
                    # Compute entropy for each pixel in batch
                    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
                    batch_outs.append(entropy)
                all_entropy = np.concatenate(batch_outs, axis=0)  # shape: (num_pixels,)
                entropies.append(all_entropy)

        # Average entropy over all classifiers
        mean_entropy = np.mean(np.stack(entropies, axis=1), axis=1)  # (num_pixels,)

        # Reshape back to image shape
        H, W = img.shape[0], img.shape[1]
        entropy_map = mean_entropy.reshape((H, W))

        # Mask zero-intensity pixels as zero entropy if desired (optional)
        mask = img.sum(axis=-1) > 0
        entropy_map[~mask] = 0.0

        if self.verbose:
            print(f"[uncertainty] Average entropy map computed: shape={entropy_map.shape}")

        # Scale the entropy map to [0, 1]
        min_val = np.min(entropy_map)
        max_val = np.max(entropy_map)
        if max_val > min_val:
            entropy_map = (entropy_map - min_val) / (max_val - min_val)
        else:
            entropy_map = np.zeros_like(entropy_map)

        return entropy_map


    def __call__(self, img):
        return self.predict(img)
