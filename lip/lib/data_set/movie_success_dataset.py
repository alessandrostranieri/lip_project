import pathlib as pl
from typing import Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np

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
        # GET THE RAW DATA
        data_dict = self.get_data(index)

        # CONVERT IT TO TENSORS

        # 1 - POSTER
        # noinspection PyTypeChecker
        X_image: Tensor = self.transform(data_dict['image'])

        # 2 - PLOT
        indexed_plot: List[int] = self.dictionary.get_indexed(data_dict['plot'])
        X_plot: Tensor = torch.tensor(indexed_plot)

        # TARGET
        budget: float = self.movie_dataset_df['budget'].iloc[index]
        revenue: float = self.movie_dataset_df['revenue'].iloc[index]

        success_label: float = float(revenue > budget)
        y: Tensor = torch.tensor([success_label])

        return X_image, X_plot, y

    def get_data(self, index: int) -> Dict[str, Any]:
        image_path: pl.Path = self.posters_path / str(self.movie_dataset_df['poster_path'].iloc[index])[1:]
        image: Image = Image.open(image_path)
        title: str = self.movie_dataset_df['title'].iloc[index]
        plot: str = self.movie_dataset_df['overview'].iloc[index]

        return {'image': image, 'title': title, 'plot': plot}


def get_class_weights(movie_data_set: MovieSuccessDataset) -> np.ndarray:
    data_set_size: int = len(movie_data_set)

    success_array: np.ndarray = np.zeros(data_set_size)

    for index, instance in enumerate(movie_data_set):
        success_array[index] = instance[2].item()

    success_array = success_array.astype(int)
    success_counts: np.ndarray = np.bincount(success_array)
    success_array = success_counts[success_array]
    success_array = 1.0 / success_array

    return success_array

# QUICK CLASS DEMO
if __name__ == '__main__':
    ms_ds: MovieSuccessDataset = MovieSuccessDataset(MOVIE_DATA_FILE,
                                                     POSTERS_DIR,
                                                     Dictionary(DATA_DIR / 'dict2000.json'))

    fig, ax = plt.subplots()

    success_index: int = 1000
    failure_index: int = 1200
    sample_index: int = success_index

    data_dict = ms_ds.get_data(sample_index)
    data_t = ms_ds[sample_index]
    success: str = 'YES' if data_t[2].item() == 1.0 else 'NO'

    ax.set_title(f"Title: {data_dict['title']}\nSuccess: {success}")
    ax.axis('off')
    ax.imshow(data_dict['image'])

    plt.show()
