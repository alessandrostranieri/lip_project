import urllib
import time
import pandas as pd
from pathlib import Path
from typing import List
from lip.utils.paths import POSTERS_DIR, DATA_DIR

# URL DIRECTORY
BASE_URL = 'http://image.tmdb.org/t/p'
POSTER_SIZE = 'w185'

if __name__ == '__main__':

    # READ METADATA FILE
    movie_metadata_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'movies_metadata.csv')
    print(f'Initial data-set consists of {movie_metadata_df.shape[0]} entries')

    # FILTER DATA WITH BUDGET
    with_budget_df: pd.DataFrame = movie_metadata_df[movie_metadata_df['budget'] != '0']
    with_revenue_df = with_budget_df[with_budget_df['revenue'] > 0.0]
    print(f'There are {with_revenue_df.shape[0]} entries with budget information')

    counter: int = 0
    posters: List[str] = []

    for index, row in with_revenue_df.iterrows():
        poster_path = row['poster_path']
        poster_url = BASE_URL + '/' + POSTER_SIZE + '/' + poster_path
        poster_destination = POSTERS_DIR / Path(poster_path).name

        if poster_path in posters:
            print(f'Already present: {poster_path} {row["title"]}')
        else:
            posters.append(poster_path)
            if not poster_destination.exists():
                urllib.request.urlretrieve(poster_url, poster_destination)
                time.sleep(0.5)
