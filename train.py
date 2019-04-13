import argparse
import os
import pickle
import pandas as pd

import tensorflow as tf
from tensorflow.python.client import device_lib

pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')  # if using Anaconda Python distro on OSX system
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from sklearn.model_selection import train_test_split

from gensim.models import KeyedVectors

import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from wcmodel.combined_model import CombinedModel
from data_loader import DataLoader

# Get root directory for this file (also root dir of full repo)
root_dir = os.path.dirname(os.path.realpath(__file__))


# Set environment variable for CUDA-enabled GPU computation
def set_visible_gpu(gpu_number: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)


# Plot loss and accuracy during training
def plot_model_accuracy(model_hist, filepath):
    # Plot training & validation accuracy values
    plt.plot(model_hist.history['acc'])
    plt.plot(model_hist.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(filepath)


def plot_model_loss(model_hist, filepath):
    # Plot training & validation loss values
    plt.plot(model_hist.history['loss'])
    plt.plot(model_hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(filepath)


# Main function for loading data, processing it, building model and starting training
def main(args):
    # Setting Keras for TF session and viewing GPU/CPU details on console
    set_visible_gpu(args.gpu)

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # K.set_session(session=sess)

    print('Machines available to TF:'+str(device_lib.list_local_devices()))
    # print('Machines available to TF through Keras:'+str(K.tensorflow_backend._get_available_gpus()))

    # ----- Model set-up, building and training -----

    # Define data-level model parameters
    max_seq_length = 256
    max_num_of_sub_sentence = 5
    max_len_of_sentence = 256

    # Load, pre-process and prepare data for feeding to model
    dl = DataLoader(max_seq_length=max_seq_length, max_num_of_sub_sentence=max_num_of_sub_sentence,
                    max_len_of_sentence=max_len_of_sentence)

    # valid_size = 40000       # set size of validation/dev set
    dl.preprocess_data()
    dl.prep_data_word_model(word_embeddings=args.word_embed)  # this will build and store vocab and embedding matrix to disk
    dl.prep_data_char_model()

    # Define model hyper-parameters (optional)


    # Load vocab and embedding matrix from disk
    with open(os.path.join(root_dir, 'models/vocab',
            (str(args.word_embed)+'_word_level_vocabulary.pickle')), 'rb') as vocab_file:
        vocabulary = pickle.load(vocab_file)

    with open(os.path.join(root_dir, 'models/embed-matrix',
                        (str(args.word_embed)+'_embedding_matrix')), 'rb') as word_embed_file:
        embeddings = np.load(word_embed_file)

    # Build and compile hybrid model
    hybrid_model = CombinedModel(max_seq_length=max_seq_length,
                                 max_num_of_sub_sentence=max_num_of_sub_sentence,
                                 max_len_of_sentence=max_len_of_sentence,
                                 wm_vocabulary=vocabulary,
                                 wm_embeddings=embeddings).build()

    # Define training configurations
    num_epochs = 25
    batch_size = 512

    # Define config for model checkpointing
    model_filepath = os.path.join(root_dir,
                                  "models/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5")

    checkpoint = ModelCheckpoint(filepath=model_filepath, monitor='val_acc', verbose=1,
                                 save_weights_only=True, mode='max')

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0,
                              mode='auto')

    callbacks_list = [earlystop, checkpoint]

    # Fit the model/ Start training
    model_hist  = hybrid_model.fit(x=[dl.X_train_cm_dict['left'], dl.X_train_cm_dict['right'],
                                      dl.X_train_wm_dict['left'], dl.X_train_wm_dict['right']],
                                   y=dl.Y_train,   epochs=num_epochs,   batch_size=batch_size,
                                   validation_data=([dl.X_validation_cm_dict['left'], dl.X_validation_cm_dict['right'],
                                                     dl.X_validation_wm_dict['left'], dl.X_validation_wm_dict['right']],
                                                    dl.Y_validation),
                                   callbacks=callbacks_list, verbose=1)

    # Save accuracy and loss plots to disk
    plot_model_accuracy(model_hist, filepath=os.path.join(root_dir, 'model-info/hybrid-model-accuracy.png'))

    plot_model_loss(model_hist, filepath=os.path.join(root_dir, 'model-info/hybrid-model-loss.png'))


if __name__ == '__main__':
    # CL argument definition
    parser = argparse.ArgumentParser()

    parser.add_argument('word_embed',
                        choices=['word2vec', 'glove', 'fasttext'],
                        help='pre-trained word embeddings to use (required)')

    parser.add_argument('--gpu',
                        default=0,
                        help='specify whether to use gpu (optional)')

    # Parse CL args
    args = parser.parse_args()

    # Setup model for training and start training
    main(args=args)


















