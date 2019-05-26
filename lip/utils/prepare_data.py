import pandas as pd

from lip.utils.common import DROP_AXIS_ROW
from lip.utils.paths import DATA_DIR, MOVIE_DATA_FILE

if __name__ == '__main__':

    # OPEN METADATA FILE
    movie_metadata_df: pd.DataFrame = pd.read_csv(DATA_DIR / 'movies_metadata.csv')

    # SELECT THE DESIRED COLUMNS
    movie_data_columns: pd.DataFrame = movie_metadata_df[['title', 'poster_path', 'overview', 'budget', 'revenue']]
    print(f'Initial data-set consists of {movie_metadata_df.shape[0]} entries')

    # FILTER BAD ENTRIES
    moview_data_columns_filtered: pd.DataFrame = movie_data_columns.dropna(axis=DROP_AXIS_ROW, inplace=False).copy()
    print(f'Drop of NaN results in {moview_data_columns_filtered.shape[0]} entries')

    moview_data_columns_filtered['budget'] = moview_data_columns_filtered['budget'].astype(float)

    moview_data_columns_filtered = moview_data_columns_filtered[moview_data_columns_filtered.budget > 0]
    print(f'Filtering of missing budget results in {moview_data_columns_filtered.shape[0]} entries')

    moview_data_columns_filtered = moview_data_columns_filtered[moview_data_columns_filtered.revenue > 0]
    print(f'Filtering of missing revenue results in {moview_data_columns_filtered.shape[0]} entries')

    moview_data_columns_filtered.to_csv(MOVIE_DATA_FILE)