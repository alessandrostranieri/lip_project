import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor

from lip.lib.data_set.dictionary import Dictionary
from lip.lib.data_set.movie_success_dataset import MovieSuccessDataset
from lip.utils.common import WORKING_IMAGE_SIDE


class PosterFeaturesNet(nn.Module):
    def __init__(self):
        super(PosterFeaturesNet, self).__init__()

        # CONVOLUTION AND DOWNSAMPLE
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # OUTPUT HAS SIX CHANNELS - THE IMAGE HAS SIDE CROPPED_IMAGE_SIZE / 2

        # INPUT MUST HAVE SIZE CHANNELS
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # OUTPUT HAS SIXTEEN CHANNELS

        self.image_side = WORKING_IMAGE_SIDE // 4  # TWO STEPS OF MAX POOL
        self.fc1 = nn.Linear(in_features=32 * self.image_side * self.image_side, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1.in_features)
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))
        return out


class PosterNet(nn.Module):
    def __init__(self, features_network: nn.Module):
        super(PosterNet, self).__init__()

        self.features_network: nn.Module = features_network
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(in_features=84, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=16)
        self.fc5 = nn.Linear(in_features=16, out_features=2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.features_network(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        y = self.softmax(x)
        return y


if __name__ == '__main__':

    from lip.utils.paths import MOVIE_DATA_FILE, POSTERS_DIR, DATA_DIR

    # CREATE A DATASET LOADER
    movie_data_set: MovieSuccessDataset = MovieSuccessDataset(MOVIE_DATA_FILE,
                                                              POSTERS_DIR,
                                                              Dictionary(DATA_DIR / 'dict2000.json'),
                                                              Compose([RandomCrop(WORKING_IMAGE_SIDE),
                                                                       ToTensor()]))
    dummy_loader: DataLoader = DataLoader(movie_data_set)

    features_network: nn.Module = PosterFeaturesNet()
    net: PosterNet = PosterNet(features_network)

    for i, (X, Xp, y) in enumerate(dummy_loader):

        if i > 0:
            break

        y_pred = net.forward(X)
        print(f'Dummy prediction: {y_pred}')
