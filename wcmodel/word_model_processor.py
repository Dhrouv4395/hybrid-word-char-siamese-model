import os
import re
import pickle
import itertools

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

import nltk
from nltk.corpus import stopwords

from gensim.models import KeyedVectors

from keras.preprocessing.sequence import pad_sequences


class WordModelDataProcessor(object):
    """
    Class for processing, cleaning and preparing data and vocabulary for word-level model.

    Attributes
    ----------
    EMBEDDING_FILE_PATH : str
        Path to the compressed pre-trained word embeddings file in disk/storage

    word2vec : gensim.models.keyedvectors.Word2VecKeyedVectors
        Pre-trained word embeddings from Google Word2Vec in gensim keyed vectors format

    stops : set
        Set of all stopwords from the nltk english stopword corpus

    """

    # Load stopwords from nltk corpus
    stops = set(stopwords.words('english'))

    # Pre-trained word embeddings available along with paths to where they are stored
    WORD_EMBED = {'word2vec': 'pre_trained_word_embeddings/GoogleNews-vectors-negative300.bin.gz',
                  'glove': 'TODO'}

    def __init__(self, train_df, valid_df, test_df, feature_cols=['question1', 'question2'],
                 word_embeddings='word2vec', embedding_dim=300, max_seq_length=256):
        """
        Parameters
        ----------
        train_df : pd.DataFrame
            training data

        test_df : pd.DataFrame
            testing data

        feature_cols : list of str
            contains names of the columns holding the data to be processed

        embedding_dim : int
            dimension of the dense pre-trained word embeddings to be used for word-level model
        """

        self.data = {'TRAIN_DF': train_df.copy(), 'VALID_DF': valid_df.copy(), 'TEST_DF': test_df.copy()}

        if word_embeddings == 'word2vec':
            self.word_embed_name = word_embeddings
            self.word_embeddings = KeyedVectors.load_word2vec_format(self.WORD_EMBED[word_embeddings], binary=True)

        # TODO
        #         else if word_embeddings == 'glove':
        #             self.word_embeddings =

        self.vocabulary = dict()
        self.inverse_vocabulary = ['<unk>']

        self.feature_cols = feature_cols
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

    @staticmethod
    def _text_to_word_list(text):
        '''
        Custom Tokenizer - Pre-process, clean and convert text/sentence to a list of words

        Parameters
        ----------
        text : str
            sentence/text from raw data which is to be tokenized

        Returns
        -------
        text : list
            processed and cleaned tokens from input text

        '''

        text = str(text)
        text = text.lower()

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        text = text.split()

        return text

    def _process_data_build_vocab(self):
        """
        Process and clean text data (sentences), build vocabulary and map cleaned text tokens
        to vocabulary indices to get [word:index] representation.
        """

        def wrd2index(text):
            t2n = []
            for word in WordModelDataProcessor._text_to_word_list(str(text)):
                # Check for unwanted words
                if word in self.stops and word not in self.word_embeddings.vocab:
                    continue

                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.inverse_vocabulary)
                    t2n.append(len(self.inverse_vocabulary))
                    self.inverse_vocabulary.append(word)
                else:
                    t2n.append(self.vocabulary[word])

            return t2n

        res_list = self.data['TRAIN_DF'][self.feature_cols[0]].apply(wrd2index)
        self.data['TRAIN_DF'][self.feature_cols[0]] = res_list
        print(self.data['TRAIN_DF'][self.feature_cols[0]])
        # self.data['TRAIN_DF'][self.feature_cols[0]] = [res[0] for res in res_list]

        res_list = self.data['TRAIN_DF'][self.feature_cols[1]].apply(wrd2index)
        self.data['TRAIN_DF'][self.feature_cols[1]] = res_list

        res_list = self.data['VALID_DF'][self.feature_cols[0]].apply(wrd2index)
        self.data['VALID_DF'][self.feature_cols[0]] = res_list

        res_list = self.data['VALID_DF'][self.feature_cols[1]].apply(wrd2index)
        self.data['VALID_DF'][self.feature_cols[1]] = res_list

        res_list = self.data['TEST_DF'][self.feature_cols[0]].apply(wrd2index)
        self.data['TEST_DF'][self.feature_cols[0]] = res_list

        res_list = self.data['TEST_DF'][self.feature_cols[1]].apply(wrd2index)
        self.data['TEST_DF'][self.feature_cols[1]] = res_list

        print(len(self.vocabulary))

        return self.vocabulary

    def _create_embed_matrix(self):
        """
        Create, initialise and build embedding matrix for the vocabulary of words.
        """

        self.embeddings = 1 * np.random.randn(len(self.vocabulary) + 1,
                                              self.embedding_dim)  # This will be the embedding matrix
        self.embeddings[0] = 0  # So that the padding will be ignored

        # Build the embedding matrix
        for word, index in self.vocabulary.items():
            if word in self.word_embeddings.vocab:
                self.embeddings[index] = self.word_embeddings.word_vec(word)

        return self.embeddings

    def _save_vocab_and_embed_matrix(self, rel_repo_path=None):
        """
        Save and persist vocabulary and embedding matrix to disk/storage.
        """

        if rel_repo_path is None:
            with open('models/vocab/{}_word_level_vocabulary.pickle'.format(self.word_embed_name), 'wb') as vocab_file:
                pickle.dump(self.vocabulary, vocab_file, protocol=pickle.HIGHEST_PROTOCOL)

            with open('models/embed-matrix/{}_embedding_matrix'.format(self.word_embed_name), 'wb') as word_embedfile:
                np.save(word_embedfile, self.embeddings)

        else:
            with open(os.path.join(rel_repo_path, '{}_word_level_vocabulary.pickle'.format(self.word_embed_name)),
                      'wb') as vocab_file:
                pickle.dump(self.vocabulary, vocab_file, protocol=pickle.HIGHEST_PROTOCOL)

            with open(os.path.join(rel_repo_path, '{}_embedding_matrix'.format(self.word_embed_name)),
                      'wb') as word_embedfile:
                np.save(word_embedfile, self.embeddings)

    def prepare_data(self):
        """
        Prepare and process data for feeding to word level model.
        """

        # Split to dicts
        self.X_train_dict = {'left': self.data['TRAIN_DF'][self.feature_cols[0]],
                             'right': self.data['TRAIN_DF'][self.feature_cols[1]]}

        self.X_validation_dict = {'left': self.data['VALID_DF'][self.feature_cols[0]],
                                  'right': self.data['VALID_DF'][self.feature_cols[1]]}

        self.test_dict = {'left': self.data['TEST_DF'][self.feature_cols[0]],
                          'right': self.data['TEST_DF'][self.feature_cols[1]]}

        # Zero padding for train and dev(validation) set
        for dataset, side in itertools.product([self.X_train_dict, self.X_validation_dict], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], maxlen=self.max_seq_length)

        # Zero padding for test set
        self.test_dict['left'] = pad_sequences(self.test_dict['left'], maxlen=self.max_seq_length)
        self.test_dict['right'] = pad_sequences(self.test_dict['right'], maxlen=self.max_seq_length)

        return self.X_train_dict, self.X_validation_dict, self.test_dict


