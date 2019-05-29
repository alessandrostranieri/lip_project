from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor

from lip.lib.data_set.dictionary import Dictionary
from lip.lib.data_set.movie_success_dataset import MovieSuccessDataset
from lip.lib.model.plot_net import PlotFeaturesNet
from lip.lib.model.poster_net import PosterFeaturesNet
from lip.utils.common import WORKING_IMAGE_SIDE


class MovieNet(nn.Module):

    def __init__(self, poster_net: nn.Module, plot_net: nn.Module):
        super(MovieNet, self).__init__()

        # THESE ARE OUR TWO NETWORKS
        self.poster_net: nn.Module = poster_net
        self.plot_net: nn.Module = plot_net
        # OUPUT IS 84 + 16 = 100
        self.fc1 = nn.Linear(in_features=100, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # GIVE GOOD NAMES TO INPUT
        x_poster: torch.Tensor = x[0]
        x_plot: torch.Tensor = x[1]

        # WE CALCULATE THE FEATURES INDEPENDENTLY
        poster_out: torch.Tensor = self.poster_net(x_poster)
        plot_out: torch.Tensor = self.plot_net(x_plot)

        # CONCATENATE
        x_concat: torch.Tensor = torch.cat((poster_out, plot_out), dim=1)

        # AND THEN SIMPLY APPLY FEED FORWARD
        x = self.fc1(x_concat)
        x = self.fc2(x)
        x = self.fc3(x)

        out = self.sigmoid(x)

        return out


# SMALL DEMO
if __name__ == '__main__':

    from lip.utils.paths import MOVIE_DATA_FILE, POSTERS_DIR, DATA_DIR

    # CREATE A DATASET LOADER
    movie_data_set: MovieSuccessDataset = MovieSuccessDataset(MOVIE_DATA_FILE,
                                                              POSTERS_DIR,
                                                              Dictionary(DATA_DIR / 'dict2000.json'),
                                                              Compose([RandomCrop(WORKING_IMAGE_SIDE),
                                                                       ToTensor()]))
    dummy_loader: DataLoader = DataLoader(movie_data_set)

    # CREATE THE NETWORK
    poster_nn: nn.Module = PosterFeaturesNet()
    plot_nn: nn.Module = PlotFeaturesNet(vocab_size=2001)
    net: MovieNet = MovieNet(poster_net=poster_nn, plot_net=plot_nn)

    # DO ONE ITERATION
    for i, (X_poster, X_plot, y) in enumerate(dummy_loader):

        if i > 0:
            break

        y_pred = net.forward((X_poster, X_plot))
        print(f'Dummy prediction: {y_pred}')
