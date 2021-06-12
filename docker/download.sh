#!/bin/bash
[ -f data/glove.840B.300d.gensim ] || {
    echo "downloading glove.840B.300d.gensim"
    gdown.pl "https://drive.google.com/file/d/1wHP5tHvBiG5aHuCHxotYQYLeQxRKI-gm/view?usp=sharing" data/glove.840B.300d.gensim
} && \
[ -f data/glove.840B.300d.gensim.vectors.npy ] || {
    echo "downloading glove.840B.300d.gensim.vectors.npy"
    gdown.pl "https://drive.google.com/file/d/1vusxVrLevVswQUAhTJEepqWGnYnV6vtY/view?usp=sharing" data/glove.840B.300d.gensim.vectors.npy
} && \
[ -f data/models/pattern_classifier.h5 ] || {
    echo "downloading pattern_classifier.h5"
    mkdir -p data/models && \
    gdown.pl "https://drive.google.com/file/d/14OtRDFrT-mAEDSqBX7dQKJGAG39xPd7c/view?usp=sharing" data/models/pattern_classifier.h5
} && \
echo "downloading spacy en_core_web_lg" && \
pip show en-core-web-lg || python -m spacy download en_core_web_lg && \
echo "downloading nltk stopwords" && \
python -c 'import nltk;nltk.download("stopwords");nltk.download("punkt");' && \
exec $@
