import pathlib as pl

DATA_DIR = pl.Path.home() / 'University' / 'data' / 'the-movies-dataset'
POSTERS_DIR = DATA_DIR / 'posters'
MOVIE_DATA_FILE: pl.Path = DATA_DIR / 'movie_data.csv'
RESULTS_DIR = DATA_DIR / 'results'

if __name__ == '__main__':
    assert POSTERS_DIR.exists(), f'{POSTERS_DIR} does not exist'
    assert DATA_DIR.exists(), f'{DATA_DIR} does not exist'
