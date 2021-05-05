from pathlib import Path
import gensim
from flair.embeddings import WordEmbeddings, TokenEmbeddings
from flair.file_utils import instance_lru_cache
import torch
from typing import List
from flair.data import Sentence
import flair
import re
import numpy as np
class MmapWordEmbeddings(WordEmbeddings):
    """
    Adapted from https://github.com/flairNLP/flair/blob/v0.8/flair/embeddings/token.py
    to use gensim with mmap
    """

    def __init__(self, embeddings: str, field: str = None):
        """
        Initializes classic word embeddings. Constructor downloads required files if not there.
        :param embeddings: one of: 'glove', 'extvec', 'crawl' or two-letter language code or custom
        If you want to use a custom embedding file, just pass the path to the embeddings as embeddings variable.
        """
        self.embeddings = embeddings

        self.instance_parameters = self.get_instance_parameters(locals=locals())

        hu_path: str = "https://flair.informatik.hu-berlin.de/resources/embeddings/token"

        cache_dir = Path("embeddings")

        # GLOVE embeddings
        if embeddings.lower() == "glove" or embeddings.lower() == "en-glove":
            cached_path(f"{hu_path}/glove.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/glove.gensim", cache_dir=cache_dir)

        # TURIAN embeddings
        elif embeddings.lower() == "turian" or embeddings.lower() == "en-turian":
            cached_path(f"{hu_path}/turian.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/turian", cache_dir=cache_dir)

        # KOMNINOS embeddings
        elif embeddings.lower() == "extvec" or embeddings.lower() == "en-extvec":
            cached_path(f"{hu_path}/extvec.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/extvec.gensim", cache_dir=cache_dir)

        # pubmed embeddings
        elif embeddings.lower() == "pubmed" or embeddings.lower() == "en-pubmed":
            cached_path(f"{hu_path}/pubmed_pmc_wiki_sg_1M.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/pubmed_pmc_wiki_sg_1M.gensim", cache_dir=cache_dir)

        # FT-CRAWL embeddings
        elif embeddings.lower() == "crawl" or embeddings.lower() == "en-crawl":
            cached_path(f"{hu_path}/en-fasttext-crawl-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/en-fasttext-crawl-300d-1M", cache_dir=cache_dir)

        # FT-CRAWL embeddings
        elif embeddings.lower() in ["news", "en-news", "en"]:
            cached_path(f"{hu_path}/en-fasttext-news-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/en-fasttext-news-300d-1M", cache_dir=cache_dir)

        # twitter embeddings
        elif embeddings.lower() in ["twitter", "en-twitter"]:
            cached_path(f"{hu_path}/twitter.gensim.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/twitter.gensim", cache_dir=cache_dir)

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 2:
            cached_path(f"{hu_path}/{embeddings}-wiki-fasttext-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/{embeddings}-wiki-fasttext-300d-1M", cache_dir=cache_dir)

        # two-letter language code wiki embeddings
        elif len(embeddings.lower()) == 7 and embeddings.endswith("-wiki"):
            cached_path(f"{hu_path}/{embeddings[:2]}-wiki-fasttext-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/{embeddings[:2]}-wiki-fasttext-300d-1M", cache_dir=cache_dir)

        # two-letter language code crawl embeddings
        elif len(embeddings.lower()) == 8 and embeddings.endswith("-crawl"):
            cached_path(f"{hu_path}/{embeddings[:2]}-crawl-fasttext-300d-1M.vectors.npy", cache_dir=cache_dir)
            embeddings = cached_path(f"{hu_path}/{embeddings[:2]}-crawl-fasttext-300d-1M", cache_dir=cache_dir)

        elif not Path(embeddings).exists():
            raise ValueError(
                f'The given embeddings "{embeddings}" is not available or is not a valid path.'
            )

        self.name: str = str(embeddings)
        self.static_embeddings = True

        if str(embeddings).endswith(".bin"):
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                str(embeddings), binary=True
            )
        else:
            self.precomputed_word_embeddings = gensim.models.KeyedVectors.load(
                str(embeddings),
                mmap='r'
            )

        self.field = field

        self.__embedding_length: int = self.precomputed_word_embeddings.vector_size
        TokenEmbeddings.__init__(self)


    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    @instance_lru_cache(maxsize=10000, typed=False)
    def get_cached_vec(self, word: str) -> torch.Tensor:
        if word in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[word]
        elif word.lower() in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[word.lower()]
        elif re.sub(r"\d", "#", word.lower()) in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[
                re.sub(r"\d", "#", word.lower())
            ]
        elif re.sub(r"\d", "0", word.lower()) in self.precomputed_word_embeddings:
            word_embedding = self.precomputed_word_embeddings[
                re.sub(r"\d", "0", word.lower())
            ]
        else:
            word_embedding = np.zeros(self.embedding_length, dtype="float")

        word_embedding = torch.tensor(
            word_embedding.tolist(), device=flair.device, dtype=torch.float
        )
        return word_embedding

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):

                if "field" not in self.__dict__ or self.field is None:
                    word = token.text
                else:
                    word = token.get_tag(self.field).value

                word_embedding = self.get_cached_vec(word=word)

                token.set_embedding(self.name, word_embedding)

        return sentences

    def __str__(self):
        return self.name

    def extra_repr(self):
        # fix serialized models
        if "embeddings" not in self.__dict__:
            self.embeddings = self.name

        return f"'{self.embeddings}'"