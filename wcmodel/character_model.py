import os
from pathlib import Path
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

from nltk.tokenize import sent_tokenize

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input,  Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D
from tensorflow.keras.layers import LSTM, Bidirectional, concatenate, Embedding
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.utils import plot_model


root_dir = Path(os.path.dirname(os.path.realpath(__file__))).parent


class CharacterModel(object):
    """
    A class for building CNN - LSTM(GRU) model for capturing morphological text features at character level.

    ...

    Atrributes
    ----------

    CHAR_DICT : str
        a string for holding dictionary of all characters - aphanumeric along with special

    """

    CHAR_DICT = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?:,\'%-\(\)/$|&;[]"'

    def __init__(self, max_len_of_sentence, max_num_of_sub_sentence, verbose=10):
        """
        Parameters
        ----------
        max_len_of_sentence : int
            Maximum length of sentence/no. of characters in sentence

        max_num_of_sub_sentence : int
            Maximum number of sub-sentences in a sentence

        verbose: int, optional
            Argument for logging data and output of intermediate ops. Default value = 10

        """

        self.max_len_of_sentence = max_len_of_sentence
        self.max_num_of_sub_sentence = max_num_of_sub_sentence
        self.verbose = verbose

        self.num_of_char = 0
        # self.num_of_label = 0
        self.unknown_label = ''

    def _build_char_dictionary(self, char_dict=None, unknown_label='UNK'):
        """
        Define possbile char set and build mapping. Using "UNK" if character does not exist in this set.

        Parameters
        ----------
        char_dict : str
            a string for holding dictionary of all characters - vocab. Default value = None

        unknown_label : str
            token string to represent OOV char. Default value = 'UNK'


        Returns
        -------
        char_indices : dict
            character to index mapping - char:ind -

        indices_char : dict
            index to character mapping - ind:char - inverse map of char_indices

        num_of_char : int
            total number of characters in vocabulary

        """

        if char_dict is None:
            char_dict = self.CHAR_DICT

        self.unknown_label = unknown_label

        chars = []

        for c in char_dict:
            chars.append(c)

        chars = list(set(chars))

        chars.insert(0, unknown_label)

        self.num_of_char = len(chars)
        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))

        if self.verbose > 5:
            print('Total number of chars:', self.num_of_char)

            print('First 3 char_indices sample:', {k: self.char_indices[k] for k in list(self.char_indices)[:3]})
            print('First 3 indices_char sample:', {k: self.indices_char[k] for k in list(self.indices_char)[:3]})

        return self.char_indices, self.indices_char, self.num_of_char

    def _transform_raw_data(self, df, x_col, sample_size=None):
        """
        Transform raw data to lists with sentences tokenized into sub-sentences.

        Parameters
        ----------
        df : pd.Dataframe
            dataframe with raw data

        x_col:
            column holding raw text data to transform

        sample_size : int
            size of dataframe (no. of rows) to transform. Default value = None


        Returns
        -------
        x : list
            transformed data of specified column with sentences tokenized into sub-sentences

        """

        x = []
        # y = []

        actual_max_sentence = 0

        if sample_size is None:
            sample_size = len(df)

        for i, row in df.head(sample_size).iterrows():
            x_data = row[x_col]
            # y_data = row[y_col]

            sentences = sent_tokenize(x_data)
            x.append(sentences)

            if len(sentences) > actual_max_sentence:
                actual_max_sentence = len(sentences)

            # y.append(label2indexes[y_data])

        if self.verbose > 5:
            print('Number of news: %d' % (len(x)))
            print('Actual max sentence: %d' % actual_max_sentence)

        return x  # return y

    def _transform_training_data(self, x_raw, max_len_of_sentence=None, max_num_of_sub_sentence=None):
        """
        Transform tokenized sentence data to 3D numpy arrays with characters mapped to indices

        Parameters
        ----------
        x_raw : list
            list of tokenized (into sub-sentences) sentences from raw data

        max_len_of_sentence : int
            maximum length of sentence/no. of characters in sentence. Default value = None

        max_num_of_sub_sentence : int
            maximum number of sub-sentences in a sentence. Default value = None


        Returns
        -------
        x : np.Array
            3D numpy array of shape - (total num of sentences, max_num_of_sub_sentence, max_len_of_sentence)
            with characters of all sub-sentences mapped to indices

        """

        unknown_value = self.char_indices[self.unknown_label]

        x = np.ones((len(x_raw), max_num_of_sub_sentence, max_len_of_sentence), dtype=np.int64) * unknown_value
        # y = np.array(y_raw)

        if max_len_of_sentence is None:
            max_len_of_sentence = self.max_len_of_sentence
        if max_num_of_sub_sentence is None:
            max_num_of_sub_sentence = self.max_num_of_sub_sentence

        for i, doc in enumerate(x_raw):
            for j, sentence in enumerate(doc):
                if j < max_num_of_sub_sentence:
                    for t, char in enumerate(sentence[-max_len_of_sentence:]):
                        if char not in self.char_indices:
                            x[i, j, (max_len_of_sentence - 1 - t)] = self.char_indices['UNK']
                        else:
                            x[i, j, (max_len_of_sentence - 1 - t)] = self.char_indices[char]

        return x  # return y

    def prepare_data(self, df, x_col,
                     max_len_of_sentence=None, max_num_of_sub_sentence=None,
                     sample_size=None):
        """
        Data processing to prepare text data for training character level CNN-LSTM model

        Parameters
        ----------
        df : pd.Dataframe
            dataframe with raw data

        x_col:
            column holding raw text data to transform

        sample_size : int
            size of dataframe (no. of rows) to transform. Default value = None

        max_len_of_sentence : int
            maximum length of sentence/no. of characters in sentence. Default value = None

        max_num_of_sub_sentence : int
            maximum number of sub-sentences in a sentence. Default value = None

        Returns
        -------
        x_processed
            processed data as a 3D numpy array ready to feed for training model

        """

        if self.verbose > 3:
            print('-----> Stage: process')

        if max_len_of_sentence is None:
            max_len_of_sentence = self.max_len_of_sentence
        if max_num_of_sub_sentence is None:
            max_num_of_sub_sentence = self.max_num_of_sub_sentence

        x_preprocessed = self._transform_raw_data(df=df, x_col=x_col, sample_size=sample_size)

        x_processed = self._transform_training_data(x_raw=x_preprocessed,
                                                    max_len_of_sentence=max_len_of_sentence,
                                                    max_num_of_sub_sentence=max_num_of_sub_sentence)

        if self.verbose > 5:
            print('Shape: ', x_processed.shape)

        return x_processed

    def _build_character_block(self, block, dropout=0.3, filters=[64, 100], kernel_size=[3, 3],
                               pool_size=[2, 2], padding='valid', activation='relu',
                               kernel_initializer='glorot_normal'):
        """
        Build block of neural network with convolutional layers to extract character level features

        Parameters
        ----------
        block : keras.layers.Embedding
            trainable Embedding layer block for character embeddings

        dropout : int
            dropout rate to use with Dropout layer for reducing over-fitting

        filters : list of int
            to determine no. of output feature maps from convolution operation with each kernel

        kernel_size : list of int
            size of 1D kernel/convolution window to use while convolution operation

        pool_size : list of int, optional
            no. of convolution layers to pool together if required

        padding : str
            mode by which to pad inputs before convolution op. Default value = 'valid'

        activation : str
            activation function to use with convolution layer. Default value = 'relu'

        kernel_initializer : str
            regularizer function applied to the kernel weights matrix. Default value = 'glorot_normal'


        Returns
        -------
        block : keras.layers.Dense
            Dense layer (fully-connected) output block after applying convolution and max pooling ops
            to input

        """

        for i in range(len(filters)):
            block = Conv1D(
                filters=filters[i], kernel_size=kernel_size[i],
                padding=padding, activation=activation, kernel_initializer=kernel_initializer)(block)

        block = Dropout(dropout)(block)
        block = MaxPooling1D(pool_size=pool_size[i])(block)

        block = GlobalMaxPool1D()(block)
        block = Dense(128, activation='relu')(block)
        return block

    def _build_sentence_block(self, max_len_of_sentence, max_num_of_sub_sentence,
                              char_dimension=16,
                              filters=[[3, 5, 7], [200, 300, 300], [300, 400, 400]],
                              # filters=[[100, 200, 200], [200, 300, 300], [300, 400, 400]],
                              kernel_sizes=[[4, 3, 3], [5, 3, 3], [6, 3, 3]],
                              pool_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                              dropout=0.4):
        """
        Model to generate and encode sentence/text with character-level embeddings using 1D convolution layers

        Parameters
        ----------
        max_len_of_sentence : int
            maximum length of sentence/no. of characters in sentence

        max_num_of_sub_sentence : int
            maximum number of sub-sentences in a sentence

        char_dimension : int
            dimension of dense character embedding. Default value = 16

        filters: list
            list of integer lists to determine sequential feature map size after convolutions op

        kernel_sizes : list
            list of integer lists with sizes of sequential convolution window to use while convolution op

        dropout : int
            dropout rate to use with Dropout layer on concatenated encoding output. Default value = 0.4


        Returns
        -------
        sent_encoder : keras.models.Model
            sentence/text encoder model to generate char-level sentence embeddings using 1D convolutional layers
            with groups of filters applied sequentially

        """

        sent_input = Input(shape=(max_len_of_sentence,), dtype='int64')
        embedded = Embedding(self.num_of_char, char_dimension, input_length=max_len_of_sentence)(sent_input)

        blocks = []
        for i, filter_layers in enumerate(filters):
            blocks.append(
                self._build_character_block(
                    block=embedded, filters=filters[i], kernel_size=kernel_sizes[i], pool_size=pool_sizes[i])
            )

        sent_output = concatenate(blocks, axis=-1)
        sent_output = Dropout(dropout)(sent_output)
        sent_encoder = Model(inputs=sent_input, outputs=sent_output, name='char_CNN_encoder')

        return sent_encoder

    def _build_char_sequence_block(self, sent_encoder, max_len_of_sentence, max_num_of_sub_sentence,
                                   dropout=0.3):
        # loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']):
        """
        Model to build sequence block with bidirectional LSTM on top of char-level sentence embeddings

        Parameters
        ----------
        sent_encoder : keras.models.Model
            char-level sentence encoder model

        max_len_of_sentence : int
            maximum length of sentence/no. of characters in sentence

        max_num_of_sub_sentence : int
            maximum number of sub-sentences in a sentence

        dropout : int
            dropout rate to use with Dropout layer on bi-LSTM output. Default value = 0.3


        Returns
        -------
        char_seq_encoder : keras.models.Model
            stacked final CNN-LSTM model to learn and extract text features at character level

        """

        char_input = Input(shape=(max_num_of_sub_sentence, max_len_of_sentence), dtype='int64')
        char_embed_output = TimeDistributed(sent_encoder, name='time_distributed_char_cnn')(char_input)

        char_seq_output = Bidirectional(LSTM(128, return_sequences=False, dropout=dropout), name='char_seq')(char_embed_output)

        char_seq_output = Dropout(dropout)(char_seq_output)

        char_seq_encoder = Model(inputs=char_input, outputs=char_seq_output, name='char_seq_encoder')

        return char_seq_encoder

    def build_model(self, char_dict=None, unknown_label='UNK', char_dimension=16,
                    save_summary=True, save_architecture_image=True, summary_filepath=None):
        """

        """

        self._build_char_dictionary(char_dict, unknown_label)

        sent_encoder = self._build_sentence_block(
            char_dimension=char_dimension,
            max_len_of_sentence=self.max_len_of_sentence, max_num_of_sub_sentence=self.max_num_of_sub_sentence)

        char_seq_model = self._build_char_sequence_block(
            sent_encoder=sent_encoder,
            max_len_of_sentence=self.max_len_of_sentence,
            max_num_of_sub_sentence=self.max_num_of_sub_sentence)

        if save_architecture_image:
            sent_encoder_model_file_path = os.path.join(root_dir,
                                            'model_info/char_feature_convolutional_model.png')

            char_model_file_path = os.path.join(root_dir, 'model_info/char_model.png')

            try:
                plot_model(sent_encoder, to_file=sent_encoder_model_file_path,
                           show_shapes=True, rankdir='TB')
                plot_model(char_seq_model, to_file=char_model_file_path, show_shapes=True,
                           rankdir='TB')
            except:
                pass

        if save_summary is True and summary_filepath is None:
            with open(os.path.join(root_dir,
                                   'model_info/hybrid_model_summary.txt'), 'w') as file_hn:
                file_hn.write("CNN-based character feature extraction model for text:\n\n")
                sent_encoder.summary(print_fn=lambda x: file_hn.write(x+"\n"))

            with open(os.path.join(root_dir,
                                   'model_info/hybrid_model_summary.txt'), 'a') as file_hn:
                file_hn.write("\n\nCharacter level sentence embedding model:\n\n")
                char_seq_model.summary(print_fn=lambda x: file_hn.write(x+"\n"))

        return char_seq_model
