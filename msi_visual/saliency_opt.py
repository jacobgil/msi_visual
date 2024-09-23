import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
from PIL import Image
import tqdm
import torchsort
import cv2

from msi_visual.percentile_ratio import TOP3
from sklearn.cluster import KMeans, kmeans_plusplus


class SaliencyOptimization:
    def __init__(
            self,
            number_of_points=500,
            regularization_strength=0.005,
            sampling="coreset",
            num_epochs=200,
            init="random",
            similarity_reg=0):
        self.num_epochs = num_epochs
        self.regularization_strength = regularization_strength
        self.sampling = sampling
        self.number_of_points = number_of_points
        self.init = init
        self.similarity_reg = similarity_reg

    def __repr__(self):
        return f"Saliency Optimization: num_epochs: {self.num_epochs} regularization_strength: {self.regularization_strength} \
            sampling: {self.sampling} number_of_points:{self.number_of_points}"

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
        D = data.shape[1]
        method = self.sampling

        if method == "random":
            return np.random.choice(list(range(N), self.number_of_points))

        elif method == "kmeans++":
            _, indices = kmeans_plusplus(data, n_clusters=Np, random_state=0)
            return indices

        elif method == "kmeans":
            kmeans = KMeans(n_clusters=Np, random_state=0, n_init="auto").fit(data)
            return pairwise_distances(data, kmeans.cluster_centers_).argmin(axis=0)

        elif method == "coreset":
            # compute mean
            u = np.mean(data, axis=0)
            # compute proposal distribution
            q = np.linalg.norm(data - u, axis=1)**2
            sum = np.sum(q)
            d = q / sum
            q = 0.5 * (d + 1.0 / N)
            # get sample and fill coreset
            return np.random.choice(N, Np, p=q)

    def set_image(self, img):
        self.img = img
        self.reshaped = self.img.reshape(
            self.img.shape[0] * self.img.shape[1], -1)
        self.img_mask = self.img.max(axis=-1) > 0
        self.resample(number_of_points=self.number_of_points)

        if isinstance(self.init, np.ndarray):
            self.visualization = torch.from_numpy(
                np.float32(self.init) / 255) * 10 - 5
            self.visualization = self.visualization.reshape(-1, 3)

        elif self.init == "random":
            self.visualization = torch.rand(
                size=(self.reshaped.shape[0], 3)) * 10 - 5

        elif self.init == "top3":
            self.visualization = torch.from_numpy(np.float32(TOP3()(img)) / 255)
            self.visualization = self.visualization.reshape(-1, 3)
        else:
            raise Exception(f"{self.init} not supported as initialization")

        if torch.cuda.is_available():
            self.visualization = self.visualization.cuda()

        if self.similarity_reg > 0:
            self.orig = self.visualization.clone()

        self.visualization.requires_grad = True
        self.optim = torch.optim.Adam([self.visualization], lr=1.0)
        delta = 0.0

        self.loss_saliency = torch.nn.MarginRankingLoss(
            margin=-delta, reduction='none')
        self.mask_np = self.reshaped.max(axis=-1) > 0
        self.mask = torch.from_numpy(self.mask_np).float().cuda()

    def resample(self, number_of_points):
        sampled_indices = self.get_reference_points(
            self.reshaped, number_of_points)
        self.indices = [
            i for i in sampled_indices if self.reshaped[i, :].max(axis=-1) > 0]

        reference_points = self.reshaped[self.indices, :]
        cosine = pairwise_distances(
            self.reshaped,
            reference_points,
            metric='cosine')
        chebyshev = pairwise_distances(
            self.reshaped, reference_points, metric='chebyshev')

        cosine = cosine.argsort().argsort()
        chebyshev = chebyshev.argsort().argsort()
        self.input_max_rank = torch.from_numpy(np.maximum(chebyshev, cosine))
        if torch.cuda.is_available():
            self.input_max_rank = self.input_max_rank.cuda()
        self.rank_squares = self.input_max_rank ** 2

    def predict(self, img):
        self.set_image(img)
        for _ in range(self.num_epochs):
            output = self.compute_epoch()
        return output

    def __call__(self, img):
        return self.predict(img)

    def compute_epoch(self):
        reference_points = self.visualization[self.indices]
        output_distances = torch.cdist(self.visualization, reference_points)
        output_ranks = torchsort.soft_rank(
            output_distances,
            regularization_strength=self.regularization_strength)

        saliency = self.loss_saliency(
            output_ranks,
            self.input_max_rank,
            torch.ones_like(output_ranks))
        saliency = (saliency * self.mask[:,
                                         None] * self.rank_squares).sum() / (self.mask[:,
                                                                                       None] * self.rank_squares).sum()

        if self.similarity_reg > 0:
            saliency = saliency + self.similarity_reg * \
                torch.nn.MSELoss()(self.visualization, self.orig)

        self.optim.zero_grad()
        loss = saliency
        loss.backward()
        self.optim.step()

        x = self.visualization.detach().cpu().numpy()
        x[self.mask_np == 0] = 0
        x = x.reshape((self.img.shape[0], self.img.shape[1], 3))

        for i in range(3):
            x[:, :, i] = x[:, :, i] - np.percentile(x[:, :, i], 0.001)
            x[:, :, i][x[:, :, i] < 0] = 0
            x[:, :, i] = x[:, :, i] / np.percentile(x[:, :, i], 99.999)
            x[:, :, i][x[:, :, i] > 1] = 1

        x = np.uint8(255 * x)

        x = cv2.cvtColor(x, cv2.COLOR_LAB2RGB)
        x[self.img_mask == 0] = 0
        return x
