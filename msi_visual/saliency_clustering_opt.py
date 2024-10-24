import numpy as np
import torch
from PIL import Image
import tqdm
import cv2
from sklearn.cluster import KMeans, kmeans_plusplus
import time
from msi_visual.utils import normalize

from msi_visual.saliency_opt import SaliencyOptimization

class SaliencyClusteringOptimization(SaliencyOptimization):
    def __init__(
            self,
            clusters = [16],
            cluster_fraction = 0.1,
            number_of_points=500,
            regularization_strength=0.005,
            sampling="coreset",
            num_epochs=200,
            init="random",
            similarity_reg=0,
            number_of_components=3,
            lab_to_rgb=True):
        super().__init__(
            number_of_points=number_of_points,
            regularization_strength=regularization_strength,
            sampling=sampling,
            num_epochs=num_epochs,
            init=init,
            similarity_reg=similarity_reg,
            number_of_components=number_of_components,
            lab_to_rgb=lab_to_rgb
        )
        self.clusters = clusters
        self.cluster_fraction = cluster_fraction
        self.visualiation_to_cluster = []
        
        for k in self.clusters:
            l = torch.nn.Sequential(torch.nn.Linear(self.number_of_components, k))
            if torch.cuda.is_available():
                l = l.cuda()
            self.visualiation_to_cluster.append(l)


    def set_image(self, img):
        super().set_image(img)
        self.cluster_labels = []
        t0 = time.time()
        for k in self.clusters:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
            # Select a random fraction of self.reshaped
            num_samples = int(self.cluster_fraction * self.reshaped.shape[0])
            random_indices = np.random.choice(self.reshaped.shape[0], num_samples, replace=False)
            sampled_data = self.reshaped[random_indices]
            kmeans.fit(sampled_data)
            labels = torch.from_numpy(kmeans.predict(self.reshaped)).long()
            if torch.cuda.is_available():
                labels = labels.cuda()
            self.cluster_labels.append(labels)
        
        print("Done with clustering", time.time() - t0)



    def get_loss(self):
        saliency_loss = super().get_loss()
        loss = None
        for layer, clusters in zip(self.visualiation_to_cluster, self.cluster_labels):
            cluster_loss = None
            cluster_loss = torch.nn.CrossEntropyLoss()(layer(self.visualization), clusters)
            if loss is None:
                loss = cluster_loss
            else:
                loss = loss + cluster_loss
        loss = loss / len(self.clusters)        
        
        #print(saliency_loss, loss)

        return 0.1*loss + saliency_loss


    def __repr__(self):
        return f"SaliencyClusteringOptimization: num_epochs: {self.num_epochs} regularization_strength: {self.regularization_strength} \
            sampling: {self.sampling} number_of_points:{self.number_of_points} clusters: {self.clusters}"

