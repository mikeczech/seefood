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
    def __init__(self, src_col, target_col):
        self._src_col = src_col
        self._target_col = target_col
        nltk.download("stopwords")
        self._nlp = spacy.load("en", disable=["parser", "ner"])
        self._stop_words = stopwords.words("english")
        self._stop_words.extend(
            [
                "from",
                "subject",
                "re",
                "edu",
                "use",
                "like",
                "recipe",
                "cook",
                "dish",
                "easy",
                "simple",
                "homemade",
                "style",
                "gluten",
                "free",
                "sweet",
                "cookbook",
                "fresh",
                "quick",
            ]
        )

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
        bigram = gensim.models.Phrases(sents, min_count=10, threshold=100)
        trigram = gensim.models.Phrases(bigram[sents], threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        nostops = self.remove_stopwords(sents)
        bigrams = self.make_bigrams(nostops, bigram_mod)
        trigrams = self.make_trigrams(bigrams, bigram_mod, trigram_mod)

        X = X.assign(**{self._target_col: self.lemmatization(trigrams)})

        return X


class UnclutterTitleTansformer(BaseEstimator, TransformerMixin):
    """
    Removes meaningless parts of titles (e.g. 'The best ...').
    """

    def __init__(self, src_col, target_col):
        self._src_col = src_col
        self._target_col = target_col

    @staticmethod
    def delete_preamble(title):
        if ":" in title:
            title_split = title.split(":")
            return title_split[-1].strip()
        return title

    @staticmethod
    def get_title_post_description(title):
        match = re.search(r".+\((.+)(\)|\.{3})", title)
        if match:
            return match.group(1)
        return title

    @staticmethod
    def remove_the_best_prefix(title):
        match = re.search(r"The Best (.+)", title)
        if match:
            return match.group(1)
        return title

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.assign(
            **{
                self._target_col: X[self._src_col].map(
                    UnclutterTitleTansformer.delete_preamble
                )
            }
        )
        X[self._target_col] = X[self._target_col].map(
            UnclutterTitleTansformer.get_title_post_description
        )
        X[self._target_col] = X[self._target_col].map(
            UnclutterTitleTansformer.remove_the_best_prefix
        )

        return X


class CreateLabelsTransformer(BaseEstimator, TransformerMixin):
    """
    Creates image labels from preprocessed titles and descriptions
    """

    def __init__(
        self, title_col, description_col, target_col, title_word_count_threshold=60
    ):
        self._title_col = title_col
        self._description_col = description_col
        self._target_col = target_col
        self._title_word_count_threshold = title_word_count_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # unclutter descriptions
        title_dict = corpora.Dictionary(X[self._title_col])
        filtered_title_vocab = [
            token
            for token in set(itertools.chain(*X[self._title_col]))
            if title_dict.dfs[title_dict.token2id[token]]
            > self._title_word_count_threshold
        ]
        uncluttered_descriptions = X[self._description_col].map(
            lambda d: [t for t in d if t in filtered_title_vocab]
        )

        # build labels
        labels = [t + d for t, d in zip(X[self._title_col], uncluttered_descriptions)]
        labels_dict = corpora.Dictionary(labels)
        high_freq_labels = [
            [t for t in l if labels_dict.dfs[labels_dict.token2id[t]] > 50]
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
