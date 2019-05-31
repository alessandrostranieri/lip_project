import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, ToTensor
from tqdm import tqdm

from lip.lib.data_set.dictionary import Dictionary
from lip.lib.data_set.movie_success_dataset import MovieSuccessDataset, get_class_weights
from lip.utils.common import WORKING_IMAGE_SIDE
from lip.utils.paths import MOVIE_DATA_FILE, POSTERS_DIR, DATA_DIR


def get_class_ratio(dataset: MovieSuccessDataset) -> float:
    success_counter: float = 0.0
    for movie_instance in tqdm(dataset):
        success_counter += movie_instance[2].item()

    return success_counter / len(dataset)


if __name__ == '__main__':

    # READ THE DATASET
    movie_data_set: MovieSuccessDataset = MovieSuccessDataset(MOVIE_DATA_FILE,
                                                              POSTERS_DIR,
                                                              Dictionary(DATA_DIR / 'dict2000.json'),
                                                              Compose([RandomCrop(WORKING_IMAGE_SIDE),
                                                                       ToTensor()]))

    print('Counting success ratio')

    # COUNT SUCCESSES
    number_of_successes: int = 0
    number_of_movies: int = len(movie_data_set)
    for i, movie_instance in enumerate(movie_data_set):
        raw_data = movie_data_set.get_data(i)
        number_of_successes += movie_instance[2].item()

    print(f'Number of successes: {number_of_successes}/{number_of_movies}')

    print(f'Accuracy when predicting always success: {number_of_successes / number_of_movies}')

    # TEST DATA SET SPLIT
    print(f'Testing success ration after split')

    data_set_size: int = len(movie_data_set)
    print(f'Size of the data-set: {data_set_size}')

    SPLIT_RATIO: float = 0.7
    training_data_set_size: int = int(data_set_size * SPLIT_RATIO)
    validation_data_set_size: int = data_set_size - training_data_set_size
    training_dataset, val_dataset = torch.utils.data.random_split(movie_data_set, [training_data_set_size,
                                                                                   validation_data_set_size])

    train_success_ratio: float = get_class_ratio(training_dataset)
    validation_success_ratio: float = get_class_ratio(val_dataset)

    print(f'Training data-set class ratio: {train_success_ratio}')
    print(f'Validation data-set class ratio: {validation_success_ratio}')

    # TEST WEIGHTED SAMPLER
    print(f'Testing weighted sampler')

    weights: np.ndarray = get_class_weights(training_dataset)

    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    weight_data_loader = DataLoader(training_dataset, batch_size=4, sampler=sampler, drop_last=True)

    # COUNT THE NUMBER OF SUCCESSES: SHOULD BE CLOSE TO 50%
    success_sum: float = 0.0
    total: float = 0.0
    draws: int = 0
    for batch in weight_data_loader:
        b = batch[2]
        total += len(b)
        success_sum += torch.sum(b)
        draws += 1

    print(f'Draws: {draws} - Success ratio (total): {success_sum / total}({total})')