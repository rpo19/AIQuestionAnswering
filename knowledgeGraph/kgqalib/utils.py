import tensorflow as tf
from transformers import DistilBertTokenizerFast
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertMainLayer

# https://www.kaggle.com/jorgemf/rnn-gru-bidirectional-attentional-context

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'W_regularizer': self.W_regularizer,
            'u_regularizer': self.u_regularizer,
            'b_regularizer': self.b_regularizer,
            'W_constraint': self.W_constraint,
            'u_constraint': self.u_constraint,
            'b_constraint': self.b_constraint,
            'bias': self.bias,
        })
        return config


def preprocess(doc):
    doc = doc.lower().replace('?', ' ?')
    return doc


class SQPBuilder():
    """
    Usage:
        # downloads distilbert
        sb = SQPCBuilder().load("/path/to/my/model.bin")

        patterns, raw_probs = sb.transform("select...")
        patterns, raw_probs = sb.transform(["select1...", "select2..."])

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
