from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor

from lip.lib.data_set.dictionary import Dictionary
from lip.lib.data_set.movie_success_dataset import MovieSuccessDataset
from lip.utils.common import WORKING_IMAGE_SIDE


class PlotNet(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()

        # EMBEDDING LAYER
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=128)

        # RECURRENT
        self.RNN = nn.GRU(input_size=128, hidden_size=128,
                          num_layers=1, batch_first=True)

        # LINEAR LAYER
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)

        r_out, r_hidden = self.RNN(x)

        # THE HIDDEN STATE HAS 3 DIMENSIONS
        # IT'S THE LAST ARRAY OF A MATRIX OF STATES
        # WE ELIMINATE THE MIDDLES DIMENSION
        x = r_hidden.contiguous().view(-1, self.RNN.hidden_size)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.sigmoid(x)

        return out


if __name__ == '__main__':

    from lip.utils.paths import MOVIE_DATA_FILE, POSTERS_DIR, DATA_DIR

    # CREATE A DATASET LOADER
    movie_data_set: MovieSuccessDataset = MovieSuccessDataset(MOVIE_DATA_FILE,
                                                              POSTERS_DIR,
                                                              Dictionary(DATA_DIR / 'dict2000.json'),
                                                              Compose([RandomCrop(WORKING_IMAGE_SIDE),
                                                                       ToTensor()]))

    dummy_loader: DataLoader = DataLoader(movie_data_set)

    m = PlotNet(vocab_size=2000)

    for i, (X, Xp, y) in enumerate(dummy_loader):

        if i > 0:
            break

        y_pred = m.forward(Xp)
        print(f'Dummy prediction: {y_pred}')

