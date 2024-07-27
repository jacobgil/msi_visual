import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
from PIL import Image
import tqdm
import torchsort
import cv2


def core_sets(data, Np):
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

    # compute mean
    u = np.mean(data, axis=0)

    # compute proposal distribution
    q = np.linalg.norm(data - u, axis=1)**2
    sum = np.sum(q)

    q2 = np.max(np.abs(data - u), axis=1)
    sum2 = np.sum(q2)

    d = q / sum + q2 / sum2
    d = d / np.sum(d)

    d = q / sum

    q = 0.5 * (d + 1.0 / N)

    # get sample and fill coreset
    samples = np.random.choice(N, Np, p=q)
    coreset = data[samples]
    weights = 1.0 / (q[samples] * Np)

    return coreset, weights, samples


class SaliencyOptimization:
    def __init__(self, number_of_points=500, regularization_strength=0.005,
                 sampling="coreset", num_epochs=200):
        self.num_epochs = num_epochs
        self.regularization_strength = regularization_strength
        self.sampling = sampling
        self.number_of_points = number_of_points

    def set_image(self, img):
        self.img = img
        self.reshaped = self.img.reshape(
            self.img.shape[0] * self.img.shape[1], -1)
        self.img_mask = self.img.max(axis=-1) > 0
        self.resample(number_of_points=self.number_of_points)
        self.visualization = torch.rand(
            size=(self.reshaped.shape[0], 3)) * 10 - 5

        if torch.cuda.is_available():
            self.visualization = self.visualization.cuda()
        self.visualization.requires_grad = True
        self.optim = torch.optim.Adam([self.visualization], lr=1.0)
        delta = 0.2 * len(self.indices)

        self.loss_saliency = torch.nn.MarginRankingLoss(
            margin=-delta, reduction='none')
        self.mask_np = self.reshaped.max(axis=-1) > 0
        self.mask = torch.from_numpy(self.mask_np).float().cuda()

    def resample(self, number_of_points):
        if self.sampling == "coreset":
            _, _, sampled_indices = core_sets(self.reshaped, number_of_points)
        elif self.sampling == "random":
            sampled_indices = np.random.choice(
                list(range(len(self.reshaped))), number_of_points)
        else:
            raise f"Sampling {self.sampling} not implemented, chose coreset/random"

        self.indices = [
            i for i in sampled_indices if self.reshaped[i, :].max(axis=-1) > 0]
        coreset = self.reshaped[self.indices, :]
        cosine = pairwise_distances(self.reshaped, coreset, metric='cosine')
        chebyshev = pairwise_distances(
            self.reshaped, coreset, metric='chebyshev')

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
