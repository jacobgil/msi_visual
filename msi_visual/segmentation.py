import numpy as np
import torch
from msi_visual.visualizations import visualizations_from_explanations
from matplotlib import pyplot as plt


class SegmentationDataset:
    def __init__(self, data, labels):
        self.data = np.float32(data)
        self.labels = np.int32(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]


class SegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(5005))
        self.model = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(5005, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),

            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),

            torch.nn.Linear(64, 4))
        print(self.model)

    def forward(self, x):
        return self.model(x * self.weights)


def load_model(model_path="models/segmentation.pth"):
    model = SegmentationModel()
    model.load_state_dict(torch.load(model_path))
    print("model loaded")
    return model


def preprocess(img):
    processed = img[:, :, : 5005]
    processed = processed / (1e-6 + np.sum(img, axis=-1)[:, :, None])
    processed = processed / \
        (1e-6 + np.percentile(processed, 99, axis=(0, 1))[None, None, :])
    processed[processed > 1] = 1
    return processed


def get_visualization(img, output):
    NUM_COMPONENTS = 5
    _cmap = plt.cm.get_cmap('gist_rainbow')
    colors_for_components = [
        np.array(
            _cmap(i)) for i in np.arange(
            0,
            1,
            1.0 /
            NUM_COMPONENTS)]

    visusalization_per_component_norm, visualization, _, _ = visualizations_from_explanations(
        img.shape, output, colors_for_components)
    return visusalization_per_component_norm, visualization


class DeepLearningSegmentation:
    def __init__(
            self,
            model_path="models/segmentation.pth",
            number_of_categories=4):
        self.model = load_model(model_path)
        self.number_of_categories = number_of_categories
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def __call__(self, img):
        processed = preprocess(img)
        vector = processed.reshape(-1, processed.shape[-1])
        input = torch.Tensor(vector)
        if torch.cuda.is_available():
            input = input.cuda()
        with torch.no_grad():
            output = self.model(input).cpu().numpy().reshape(
                img.shape[0], img.shape[1], self.number_of_categories)
        output = output.transpose((2, 0, 1))
        return output
