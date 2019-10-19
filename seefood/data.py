import re
import os
import itertools

import torch
from torchvision import transforms

from PIL import Image

import gensim
from gensim import corpora
import spacy
import nltk
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class ThumbnailUrlTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, src_col, target_col, image_base_path):
        self._src_col = src_col
        self._target_col = target_col
        self._image_base_path = image_base_path

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.assign(
            **{
                self._target_col: X[self._src_col].apply(
                    lambda x: os.path.join(self._image_base_path, x.rsplit("/", 1)[-1])
                )
            }
        )

        X = X[X[self._target_col].apply(lambda x: os.path.isfile(x))]

        return X


class BasicTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, src_col, target_col, bigram_threshold=50, trigram_threshold=50):
        self._src_col = src_col
        self._target_col = target_col
        nltk.download("stopwords")
        self._nlp = spacy.load("en_core_web_md")
        self._stop_words = stopwords.words("english")
        self._bigram_threshold = bigram_threshold
        self._trigram_threshold = trigram_threshold

    def remove_stopwords(self, sents):
        return [
            [
                w
                for w in gensim.utils.simple_preprocess(str(s))
                if w not in self._stop_words
            ]
            for s in sents
        ]

    def make_bigrams(self, sents, bigram_mod):
        return [bigram_mod[s] for s in sents]

    def make_trigrams(self, sents, bigram_mod, trigram_mod):
        return [trigram_mod[bigram_mod[s]] for s in sents]

    def lemmatization(self, sents, allowed_postags=["NOUN"]):
        sents_out = []
        for s in sents:
            doc = self._nlp(" ".join(s))
            sents_out.append(
                [token.lemma_ for token in doc if token.pos_ in allowed_postags]
            )
        return sents_out

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sents = X[self._src_col].apply(
            lambda x: gensim.utils.simple_preprocess(str(x), deacc=True)
        )
        bigram = gensim.models.Phrases(sents, min_count=10, threshold=self._bigram_threshold)
        trigram = gensim.models.Phrases(bigram[sents], threshold=self._trigram_threshold)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        nostops = self.remove_stopwords(sents)
        bigrams = self.make_bigrams(nostops, bigram_mod)
        trigrams = self.make_trigrams(bigrams, bigram_mod, trigram_mod)

        X = X.assign(**{self._target_col: self.lemmatization(trigrams)})

        return X


class CreateLabelsTransformer(BaseEstimator, TransformerMixin):
    """
    Creates image labels from preprocessed titles and descriptions
    """

    def __init__(
        self, title_col, description_col, target_col, high_freq_threshold=60
    ):
        self._title_col = title_col
        self._description_col = description_col
        self._target_col = target_col
        self._high_freq_threshold = high_freq_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # unclutter descriptions
        title_vocab = set(itertools.chain(*X[self._title_col]))
        uncluttered_descriptions = X[self._description_col].map(
            lambda d: [t for t in d if t in title_vocab]
        )

        # build labels
        labels = [t + d for t, d in zip(X[self._title_col], uncluttered_descriptions)]
        labels_dict = corpora.Dictionary(labels)
        high_freq_labels = [
            [t for t in l if labels_dict.dfs[labels_dict.token2id[t]] > self._high_freq_threshold]
            for l in labels
        ]
        X = X.assign(**{self._target_col: [set(sent) for sent in high_freq_labels]})

        return X


class SelectColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self._columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._columns]


class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        return self.df.iloc[idx]
