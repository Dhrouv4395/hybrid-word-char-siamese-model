import os
import pandas as pd
from sklearn.model_selection import train_test_split

from wcmodel.word_model_processor import WordModelDataProcessor
from wcmodel.character_model import CharacterModel


class DataLoader(object):
    """
    Class for loading training and testing data from storage and pre-processing it to feed for training
    hybrid word and character model.

    """

    def __init__(self, max_seq_length=256, max_num_of_sub_sentence=5, max_len_of_sentence=256,
                 repo_path=None, feature_cols=['question1', 'question2'],
                 address=None, port=None):
        """
        Parameters
        ----------
        address: str
            the address of the S3 instance where training data will be stored

        port : ints
            the port on which access is open to the S3 instance

        repo_path : str
            the full path to the repository holding the training and testing data on the S3 instance
        """

        if address is not None and port is not None and repo_path is not None:
            self.data_repo_path = os.path.join("{}:{}".format(address, port), repo_path)

        if repo_path is not None:
            self.data_repo_path = repo_path
        else:
            self.data_repo_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                               'data')

        self.train_data = pd.read_csv(os.path.join(self.data_repo_path, 'train.csv'))
        self.test_data = pd.read_csv(os.path.join(self.data_repo_path, 'test.csv'))

        self.feature_cols = feature_cols

        self.max_seq_length = max_seq_length
        self.max_num_of_sub_sentence = max_num_of_sub_sentence
        self.max_len_of_sentence = max_len_of_sentence

    def preprocess_data(self, valid_size=40000):
        """
        Basic pre-processing and splitting to train data to train and dev sets.
        :rtype: object
        """

        self.train_data = self.train_data.dropna()
        self.test_data = self.test_data.dropna()

        self.training_size = len(self.train_data) - valid_size

        self.X = self.train_data[self.feature_cols]
        self.Y = self.train_data['is_duplicate']

        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X, self.Y,
                                                                                            test_size=valid_size)

        # Convert labels to their numpy representations
        self.Y_train = self.Y_train.values
        self.Y_validation = self.Y_validation.values

    def prep_data_word_model(self, word_embeddings='word2vec'):
        """
        Prepare data for the word model.
        """

        wmdp = WordModelDataProcessor(self.X_train, self.X_validation, self.test_data,
                                      word_embeddings=word_embeddings, feature_cols=self.feature_cols,
                                      max_seq_length=self.max_seq_length)

        wmdp._process_data_build_vocab()
        wmdp._create_embed_matrix()
        wmdp._save_vocab_and_embed_matrix()

        self.X_train_wm_dict, self.X_validation_wm_dict, self.test_wm_dict = wmdp.prepare_data()

        return self.X_train_wm_dict, self.X_validation_wm_dict, self.test_wm_dict

    def prep_data_char_model(self, max_len_of_sentence=256, max_num_of_sub_sentence=5):
        """
        Prepare data for the character model.
        """

        if max_len_of_sentence is None:
            max_len_of_sentence = self.max_len_of_sentence
        if max_num_of_sub_sentence is None:
            max_num_of_sub_sentence = self.max_num_of_sub_sentence

        cmdp = CharacterModel(max_len_of_sentence=max_len_of_sentence,
                              max_num_of_sub_sentence=max_num_of_sub_sentence)

        cmdp._build_char_dictionary()

        self.X_train_cm_dict = {}
        self.X_validation_cm_dict = {}
        self.test_cm_dict = {}

        self.X_train_cm_dict.update({'left': cmdp.prepare_data(self.X_train, x_col=self.feature_cols[0])})
        self.X_train_cm_dict.update({'right': cmdp.prepare_data(self.X_train, x_col=self.feature_cols[1])})

        self.X_validation_cm_dict.update({'left': cmdp.prepare_data(self.X_validation,
                                                                    x_col=self.feature_cols[0])})
        self.X_validation_cm_dict.update({'right': cmdp.prepare_data(self.X_validation,
                                                                     x_col=self.feature_cols[1])})

        self.test_cm_dict.update({'left': cmdp.prepare_data(self.test_data, x_col=self.feature_cols[0])})
        self.test_cm_dict.update({'right': cmdp.prepare_data(self.test_data, x_col=self.feature_cols[1])})

        return self.X_train_cm_dict, self.X_validation_cm_dict, self.test_cm_dict
