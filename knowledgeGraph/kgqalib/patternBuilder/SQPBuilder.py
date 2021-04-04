import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertMainLayer
from shared.attention import AttentionWithContext


class SQPBuilder():
    """
    Usage:
        # downloads distilbert
        sb = SQPBuilder().load("/path/to/my/model.bin")

        patterns, raw_probs = sb.transform("who is...")
        patterns, raw_probs = sb.transform(["what are...", "how many..."])

        #####

        (
            ['p1', 'p2'],
            array(
                [
                    [5.8642190e-06, 9.8887116e-01, 4.9565523e-03, 2.1186814e-05,
                        6.0184798e-03, 2.7468861e-05, 9.5135392e-06, 2.7063636e-05,
                        2.2107159e-05, 1.4005833e-05, 1.7572796e-05, 9.0330796e-06],
                    [5.8642190e-06, 9.8887116e-01, 4.9565523e-03, 2.1186834e-05,
                        6.0184798e-03, 2.7468861e-05, 9.5135392e-06, 2.7063636e-05,
                        2.2107159e-05, 1.4005833e-05, 1.7572796e-05, 9.0330796e-06]
                ],
                dtype=float32)
        )

    """

    def __init__(self):
        self.model = None
        self.MAX_LENGTH = 25
        self.CATEGORIES = [
            'p0',
            'p1',
            'p2',
            'p3',
            'p4',
            'p5',
            'p6',
            'p7',
            'p8',
            'p9',
            'p10',
            'p_notFound'
        ]

    def load(self, modelPath):
        print('Loading pattern classifier...')
        self.model = tf.keras.models.load_model(modelPath, custom_objects={
            'AttentionWithContext': AttentionWithContext,
            'TFDistilBertMainLayer': TFDistilBertMainLayer
        })
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased")
        return self

    def __preprocess(self, text):
        return text.lower().replace('?', ' ?')

    def __bertTokenize(self, docs):
        input_ids = []
        attention_masks = []
        if type(docs) == str:
            docs = [docs]
        for doc in docs:
            bert_inp = self.tokenizer.encode_plus(
                doc, add_special_tokens=True,  max_length=self.MAX_LENGTH,
                padding='max_length', return_attention_mask=True, truncation=True)
            input_ids.append(bert_inp['input_ids'])
            attention_masks.append(bert_inp['attention_mask'])
        return np.array(input_ids, dtype='int32'), np.array(attention_masks, dtype='int32')

    def transform(self, queries):
        input_ids, attention_masks = self.__bertTokenize(queries)
        raw_predictions = self.model.predict([input_ids, attention_masks])
        most_likely_patterns = list(
            map(lambda x: self.CATEGORIES[x.argmax()], raw_predictions))
        return most_likely_patterns, raw_predictions