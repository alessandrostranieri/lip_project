import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor

from lip.lib.data_set.dictionary import Dictionary
from lip.lib.data_set.movie_success_dataset import MovieSuccessDataset
from lip.utils.common import WORKING_IMAGE_SIDE


class PlotFeaturesNet(nn.Module):
    def __init__(self, vocab_size: int):

        super(PlotFeaturesNet, self).__init__()

        # EMBEDDING LAYER
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128)

        # RECURRENT
        self.RNN = nn.GRU(input_size=128, hidden_size=128,
                          num_layers=1, batch_first=True)

        # LINEAR LAYER
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)

        r_out, r_hidden = self.RNN(x)

        # THE HIDDEN STATE HAS 3 DIMENSIONS
        # IT'S THE LAST ARRAY OF A MATRIX OF STATES
        # WE ELIMINATE THE MIDDLES DIMENSION
        x = r_hidden.contiguous().view(-1, self.RNN.hidden_size)

        x = self.fc1(x)
        out = self.fc2(x)

        return out


class PlotNet(nn.Module):

    def __init__(self, features_network: nn.Module):
        super(PlotNet, self).__init__()

        self.features_network: nn.Module = features_network

        self.fc1 = nn.Linear(16, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_network(x)
        x = self.fc1(x)
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

    features_nn: nn.Module = PlotFeaturesNet(vocab_size=2001)
    net: PlotNet = PlotNet(features_nn)

    for i, (X, Xp, y) in enumerate(dummy_loader):

        if i > 0:
            break

        y_pred = net.forward(Xp)
        print(f'Dummy prediction: {y_pred}')
