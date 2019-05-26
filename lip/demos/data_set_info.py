import torch
from torchvision.transforms import Compose, RandomCrop, ToTensor
from tqdm import tqdm

from lip.lib.data_set.movie_success_dataset import MovieSuccessDataset
from lip.utils.common import CROPPED_IMAGE_SIDE
from lip.utils.paths import MOVIE_DATA_FILE, POSTERS_DIR

if __name__ == '__main__':

    # READ THE DATASET
    movie_data_set: MovieSuccessDataset = MovieSuccessDataset(MOVIE_DATA_FILE,
                                                              POSTERS_DIR,
                                                              Compose([RandomCrop(CROPPED_IMAGE_SIDE),
                                                                       ToTensor()]))

    # COUNT SUCCESSES
    number_of_successes: int = 0
    number_of_movies: int = len(movie_data_set)
    for sample in tqdm(movie_data_set):

        X, y = sample

        y_t: torch.Tensor = y
        number_of_successes += int(y_t[0])

    print(f'Number of successes: {number_of_successes}/{number_of_movies}')