import os
from pathlib import Path
import pandas as pd
pd.options.mode.chained_assignment = None

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Lambda, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import plot_model

from wcmodel.character_model import CharacterModel


root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent


class CombinedModel(object):
    """
    Class for building combined word and character model in a Siamese architecture.

    """

    def __init__(self, max_seq_length, max_num_of_sub_sentence, max_len_of_sentence,
                wm_vocabulary, wm_embeddings,
                wm_lstm_hidden_units=50, wm_lstm_dropout=0.2, wm_embedding_dim=300,
                cm_char_dict=None, cm_unknown_label='UNK', cm_char_dimension=16,
                gradient_clip_norm=1.25, loss='mean_squared_error', metrics=['accuracy']):
        """
        Initialise constructor with parameters and hyperparameters for final combined word and char model.

        Parameters
        ----------

        """

        self.max_seq_length = max_seq_length
        self.max_num_of_sub_sentence = max_num_of_sub_sentence
        self.max_len_of_sentence = max_len_of_sentence

        self.wm_embeddings = wm_embeddings
        self.wm_vocabulary = wm_vocabulary
        self.wm_lstm_hidden_units = wm_lstm_hidden_units
        self.wm_embedding_dim = wm_embedding_dim
        self.wm_lstm_dropout = wm_lstm_dropout

        self.cm_char_dict = cm_char_dict
        self.cm_unknown_label = cm_unknown_label
        self.cm_char_dimension = cm_char_dimension

        self.gradient_clip_norm = gradient_clip_norm
        self.loss = loss
        self.metrics = metrics

    class Similarity:
        """
        A (inner) class for implementing common distance/similarity measures.
        """

        def __init__(self, left, right):
            """
            Initialise both tensors for which similarity needs to be computed.
            """
            self.left = left
            self.right = right

        def euclidean_similarity(self):
            """
            Euclidean similarity implementation
            """

            return K.sqrt(K.sum(K.square(self.left - self.right), axis=1, keepdims=True))

        def manhattan_similarity(self):
            """
            Manhattan similarity implementation
            """

            return K.exp(-K.sum(K.abs(self.left - self.right), axis=1, keepdims=True))

    def build(self, save_summary=True, save_architecture_image=True):
        """
        Build the combined (hybrid) word and character model.

        Returns
        -------
        hybrid_model: keras.models.Model
            hybrid character and word final model

        """

        left_char_input = Input(shape=(self.max_num_of_sub_sentence, self.max_len_of_sentence,), dtype='int64',
                                name='left_char_input')
        right_char_input = Input(shape=(self.max_num_of_sub_sentence, self.max_len_of_sentence,), dtype='int64',
                                 name='right_char_input')

        left_word_input = Input(shape=(self.max_seq_length,), dtype='int64', name='left_word_input')
        right_word_input = Input(shape=(self.max_seq_length,), dtype='int64', name='right_word_input')

        word_embed_layer = Embedding(input_dim=len(self.wm_embeddings), output_dim=self.wm_embedding_dim,
                                                     weights=[self.wm_embeddings], input_length=self.max_seq_length,
                                                     trainable=False)

        word_encoded_left = word_embed_layer(left_word_input)
        word_encoded_right = word_embed_layer(right_word_input)

        word_bi_gru = Bidirectional(GRU(units=self.wm_lstm_hidden_units, return_sequences=False,
                                        dropout=self.wm_lstm_dropout))

        word_fv_left = word_bi_gru(word_encoded_left)
        word_fv_right = word_bi_gru(word_encoded_right)

        cm = CharacterModel(max_len_of_sentence=self.max_len_of_sentence,
                            max_num_of_sub_sentence=self.max_num_of_sub_sentence)

        char_seq_model = cm.build_model(char_dict=self.cm_char_dict, unknown_label=self.cm_unknown_label,
                                        char_dimension=self.cm_char_dimension, save_summary=True)

        char_fv_left = char_seq_model(left_char_input)
        char_fv_right = char_seq_model(right_char_input)

        concat_fv_left = keras.layers.concatenate([word_fv_left, char_fv_left], axis=-1)
        concat_fv_right = keras.layers.concatenate([word_fv_right, char_fv_right], axis=-1)

        similarity = Lambda(function=lambda x: CombinedModel.Similarity(x[0], x[1]).manhattan_similarity(),
                            output_shape=lambda x: (x[0][0], 1))([concat_fv_left, concat_fv_right])

        hybrid_model = Model([left_char_input, right_char_input, left_word_input, right_word_input],
                             [similarity])

        optimizer = Adadelta(clipnorm=self.gradient_clip_norm)

        hybrid_model.compile(loss=self.loss, optimizer=optimizer, metrics=self.metrics)

        if save_summary:
            with open(os.path.join(root_dir,
                                   'model_info/hybrid_model_summary.txt'), 'a') as file_hn:
                file_hn.write("\n\nFinal hybrid word plus character siamese model:\n\n")
                hybrid_model.summary(print_fn=lambda x: file_hn.write(x+"\n"))

        if save_architecture_image:
            hybrid_model_filepath = os.path.join(root_dir, 'model_info/hybrid_final_model.png')
            try:
                plot_model(hybrid_model, to_file=hybrid_model_filepath,
                           show_shapes=True, rankdir='LR')
            except:
                pass

        return hybrid_model


