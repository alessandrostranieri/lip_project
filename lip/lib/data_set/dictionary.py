import pathlib as pl
import json
from typing import Dict, List
import pandas as pd

from nltk import WordNetLemmatizer

from lip.utils.paths import DATA_DIR, MOVIE_DATA_FILE
import nltk


class Dictionary(object):

    def __init__(self, dictionary_file: pl.Path, indexed_size: int = 50) -> None:
        super().__init__()
        assert dictionary_file.exists(), f'Dictionary file {dictionary_file} not found'
        with open(dictionary_file) as f:
            self.dictionary: Dict[str, int] = json.load(f)
        print(f'Loaded dictionary with {len(self.dictionary) - 1} elements')  # -1 because of UNK

        self.indexed_size: int = indexed_size
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()

    def get_indexed(self, sentence: str) -> List[int]:
        tokenized_sentence: List[str] = nltk.word_tokenize(sentence)
        tokenized_sentence = tokenized_sentence[:self.indexed_size]
        result: List[int] = [0] * 50
        for index, token in enumerate(tokenized_sentence):
            lemma: str = self.lemmatizer.lemmatize(token)
            result[index] = self.dictionary.get(lemma, 0)

        return result


# DEMO
if __name__ == '__main__':
    # OPEN DICTIONARY
    dictionary: Dictionary = Dictionary(DATA_DIR / 'dict2000.json')
    # OPEN MOVIE METADATA
    movie_meta_data_df: pd.DataFrame = pd.read_csv(MOVIE_DATA_FILE)

    dummy_plot: str = movie_meta_data_df['overview'].iloc[42]
    print(f'Original plot: {dummy_plot}')

    indexed_plot: List[int] = dictionary.get_indexed(dummy_plot)
    print(indexed_plot)

