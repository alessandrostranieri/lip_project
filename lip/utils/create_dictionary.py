import json
from typing import List, Set, Dict

import nltk
import pandas as pd
from nltk import FreqDist
from nltk import WordNetLemmatizer, RegexpTokenizer
from nltk.corpus import stopwords

from lip.utils.paths import MOVIE_DATA_FILE, DATA_DIR

# CONSTANTS
NUM_MOST_FREQUENT: int = 2000

movies_df: pd.DataFrame = pd.read_csv(MOVIE_DATA_FILE)

plots: List[str] = movies_df['overview'].tolist()
print(f'The longest plot has length: {len(max(plots, key=len))}')
print(f'The shortest plot has length: {len(min(plots, key=len))}')

print(f'We collected {len(plots)} movie plots')

raw_tokens: Set[str] = set()

for plot in plots:
    plot_words = nltk.word_tokenize(plot)
    raw_tokens.update(plot_words)

print(f'The initial list of words contains {len(raw_tokens)} distinct elements')

print(f'Let\'s do some filtering...')

# LEMMATIZER
wordnet_lemmatizer = WordNetLemmatizer()
# STOPWORDS
english_stopwords = stopwords.words('english')
# WE USE THIS TO REMOVE PUNCTUATION
only_words_tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+')

nltk_lemmas_list: List[str] = []

for plot in plots:
    plot_tokens: List[str] = only_words_tokenizer.tokenize(plot)
    for plot_token in plot_tokens:
        lower_case_token: str = plot_token.lower()
        lemma: str = wordnet_lemmatizer.lemmatize(lower_case_token)
        if lemma not in english_stopwords:
            nltk_lemmas_list.append(lemma)

nltk_lemmas_set: Set[str] = set(nltk_lemmas_list)

print(f'Using WordNet lemmatizer we obtained: {len(nltk_lemmas_set)}')

print(f'We will take only the {NUM_MOST_FREQUENT} most frequent words')

frequency_distribution: FreqDist = FreqDist(nltk_lemmas_list)
most_frequent_words = frequency_distribution.most_common(NUM_MOST_FREQUENT)

print(most_frequent_words)

dictionary: Dict[str, int] = {'UNK': 0}

for index, frequent_word in enumerate(most_frequent_words, 1):
    dictionary[frequent_word[0]] = index

with open(DATA_DIR / f'dict{NUM_MOST_FREQUENT}.json', 'w') as f:
    json.dump(dictionary, f, indent=2, ensure_ascii=True)