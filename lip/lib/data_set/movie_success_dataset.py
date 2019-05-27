import pathlib as pl
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from lip.lib.data_set.dictionary import Dictionary
from lip.utils.paths import POSTERS_DIR, MOVIE_DATA_FILE, DATA_DIR

DS_IMAGE_INDEX: int = 0
DS_TITLE_INDEX: int = 1
DS_PLOT_INDEX: int = 2


class MovieSuccessDataset(Dataset):
    """Represents the movie dataset"""

    def __init__(self,
                 movie_dataset_csv: pl.Path,
                 posters_path: pl.Path,
                 dictionary: Dictionary,
                 transform=transforms.Compose([transforms.ToTensor()])):
        assert movie_dataset_csv.exists(), f'Movie dataset file {movie_dataset_csv} does not exist'
        assert posters_path.exists(), f'Posters path {posters_path} does not exist'

        self.movie_dataset_df: pd.DataFrame = pd.read_csv(movie_dataset_csv)
        self.posters_path: pl.Path = posters_path
        self.dictionary: Dictionary = dictionary
        self.transform = transform

    def __len__(self) -> int:
        return len(self.movie_dataset_df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path: pl.Path = self.posters_path / str(self.movie_dataset_df['poster_path'].iloc[index])[1:]
        # image: np.ndarray = io.imread(str(image_path))
        image: Image = Image.open(image_path)

        # noinspection PyTypeChecker
        transformed_image: Tensor = self.transform(image)

        title: str = self.movie_dataset_df['title'].iloc[index]

        plot: str = self.movie_dataset_df['overview'].iloc[index]
        indexed_plot: List[int] = self.dictionary.get_indexed(plot)

        budget: float = self.movie_dataset_df['budget'].iloc[index]
        revenue: float = self.movie_dataset_df['revenue'].iloc[index]

        success_label: float = float((revenue / budget) >= 1.0)

        X_image: Tensor = transformed_image
        X_plot: Tensor = torch.tensor(indexed_plot)
        y: Tensor = torch.tensor([success_label])

        return X_image, X_plot, y


# QUICK CLASS DEMO
if __name__ == '__main__':
    ms_ds: MovieSuccessDataset = MovieSuccessDataset(MOVIE_DATA_FILE,
                                                     POSTERS_DIR,
                                                     Dictionary(DATA_DIR / 'dict2000.json'))

    fig, ax = plt.subplots()

    sample_index: int = 100
    sample = ms_ds[sample_index]

    sample_image: np.ndarray = sample[0].numpy().transpose((1, 2, 0))
    sample_plot: torch.Tensor = sample[1]
    print(sample_plot)

    ax.axis('off')
    ax.imshow(sample_image)

    plt.show()
