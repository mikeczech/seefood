import re
import os

import torch
from torchvision import transforms

from PIL import Image

import gensim
import spacy
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class ImageRead(BaseEstimator, TransformerMixin):
    def __init__(self, image_base_path, image_size):
        self.image_base_path = image_base_path
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        X["thumbnail"] = X["thumbnail"].apply(
            lambda x: os.path.join(self.image_base_path, x.rsplit("/", 1)[-1])
        )
        X = X[X["thumbnail"].apply(lambda x: os.path.isfile(x))]

        X["thumbnail"] = X["thumbnail"].apply(
            lambda img_path: Image.open(img_path).convert("RGB")
        )
        X["thumbnail"] = X["thumbnail"].apply(lambda img: self.transform_img(img))

        return X


class PreprocessText(BaseEstimator, TransformerMixin):
    def __init__(self, col_name):
        self._col_name = col_name
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

    def lemmatization(self, sents):
        sents_out = []
        for s in sents:
            doc = self._nlp(" ".join(s))
            sents_out.append([token.lemma_ for token in doc])
        return sents_out

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        sents = X[self._col_name].apply(
            lambda x: gensim.utils.simple_preprocess(x, deacc=True)
        )
        bigram = gensim.models.Phrases(sents, min_count=10, threshold=100)
        trigram = gensim.models.Phrases(bigram[sents], threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        nostops = self.remove_stopwords(sents)
        bigrams = self.make_bigrams(nostops, bigram_mod)
        trigrams = self.make_trigrams(
            bigrams, bigram_mod, trigram_mod
        )
        X[self._col_name] = self.lemmatization(trigrams)

        return X


class TitleTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

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
        pass

    def transform(self, X, y=None):
        X["title"] = X["title"].map(TitleTransform.delete_preamble)
        X["title"] = X["title"].map(TitleTransform.get_title_post_description)
        X["title"] = X["title"].map(TitleTransform.remove_the_best_prefix)

        return X


class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        return self.df.iloc[idx]
