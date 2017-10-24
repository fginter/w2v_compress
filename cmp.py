import six
assert six.PY3, "run me with python3"

import json
import traceback
import lwvlib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import keras
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Conv1D, Concatenate
from keras.layers.pooling import MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.initializers import Zeros


from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge
from keras.layers.core import Masking, Flatten, Lambda
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import Callback,ModelCheckpoint
import pickle
import sys
from keras.utils import plot_model, normalize
import keras.backend as K

def get_char_sequences(words,char_dict,max_seq_len,learn_dict):
    """
    Returns a matrix of character index sequences
    """
    char_sequences=np.zeros((len(words),max_seq_len),dtype=np.int32) #embedding indices
    for i,word in enumerate(words):
        word=["WORDBEG"]+list(word)+["WORDEND"]
        word=word[-max_seq_len:]
        for j,c in enumerate(word):
            if learn_dict:
                char_sequences[i,j]=char_dict.setdefault(c,len(char_dict))
            else:
                char_sequences[i,j]=char_dict.get(c,0)
    return char_sequences

def get_data(wv,args):
    """Data for training for the wv model"""
    if args.chardict:
        print("Loading chardict", args.chardict, file=sys.stderr)
        with open(args.chardict) as f:
            char_dict=json.load(f)
    else:
        char_dict={"MASK":0,"WORDBEG":1,"WORDEND":2}
    char_sequences=get_char_sequences(wv.words,char_dict,args.max_seq_len,learn_dict=True)
    return char_dict, char_sequences

def net_5lstm(wv_model,args):
    inp_chars=Input(shape=(args.max_seq_len,), name="chars", dtype='int32')
    inp_chars_embed=Embedding(input_dim=len(char_dict),output_dim=args.charemb,input_length=args.max_seq_len,mask_zero=True,embeddings_initializer=Zeros(),name="charemb")(inp_chars)
    lstm_lr=Bidirectional(LSTM(units=wv_model.vectors.shape[1],name="lstmlr",activation="tanh",recurrent_activation="hard_sigmoid",return_sequences=True,dropout=args.dropout))(inp_chars_embed)
    lstm_1=LSTM(units=wv_model.vectors.shape[1],name="lstm1",activation="tanh",return_sequences=True)(lstm_lr)
    lstm_2_in=Concatenate()([lstm_1, lstm_lr])
    lstm_2=LSTM(units=wv_model.vectors.shape[1],name="lstm2",activation="tanh",return_sequences=True)(lstm_2_in)
    lstm_3_in=Concatenate()([lstm_2, lstm_1])
    lstm_3=LSTM(units=wv_model.vectors.shape[1],name="lstm3",activation="tanh",return_sequences=True)(lstm_3_in)
    lstm_4_in=Concatenate()([lstm_2, lstm_3])
    lstm_4=LSTM(units=wv_model.vectors.shape[1],name="lstm4",activation="tanh",return_sequences=True)(lstm_4_in)
    lstm_5_in=Concatenate()([lstm_4, lstm_3, lstm_lr])
    lstm_5=LSTM(units=wv_model.vectors.shape[1],name="lstm5",activation="tanh")(lstm_5_in)
    #nonlin=Dense(wv_model.vectors.shape[1], activation="tanh", name="nonlin1")(lstm_1)
    #l2_norm = Lambda(lambda  x: K.l2_normalize(x,axis=-1))(lstm_5)
    lin_out=Dense(wv_model.vectors.shape[1],name="linout")(lstm_5)
    return inp_chars,lin_out

def net_1lstm(wv_model,args):
    inp_chars=Input(shape=(args.max_seq_len,), name="chars", dtype='int32')
    inp_chars_embed=Embedding(input_dim=len(char_dict),output_dim=args.charemb,input_length=args.max_seq_len,mask_zero=True,embeddings_initializer=Zeros(),name="charemb")(inp_chars)
    lstm_lr=Bidirectional(LSTM(units=wv_model.vectors.shape[1],name="lstmlr",activation="tanh",recurrent_activation="hard_sigmoid",return_sequences=False,dropout=args.dropout))(inp_chars_embed)
    lin_out=Dense(wv_model.vectors.shape[1],name="linout")(lstm_lr)
    return inp_chars,lin_out


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sclassifier')
    parser.add_argument('--embeddings', help='embeddings file')
    parser.add_argument('--edim', type=int, default=200000, help='Embedding max_rank. Default %(default)d')
    parser.add_argument('--max-seq-len', type=int, default=20, help='Max word length. Default %(default)d')
    parser.add_argument('--charemb', type=int, default=200, help='Character embedding length. Default %(default)d')
    parser.add_argument('--model-base', default='model-5lstm',help="Base name for the model file")
    parser.add_argument('--chardict',help="Read this chardict")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout. Default %(default)f")
    parser.add_argument('--lr', type=float, default=0.005, help="Learning rate. Default %(default)f")
    parser.add_argument('--epochs', type=int, default=30, help="Epochs. Default %(default)d")
    
    args = parser.parse_args()

    #source of embeddings
    wv_model=lwvlib.load(args.embeddings,args.edim,args.edim)
    char_dict,char_sequences=get_data(wv_model,args)
    print("Char dict len", len(char_dict), file=sys.stderr)
    print("Char sequences shape", char_sequences.shape, file=sys.stderr)
    
    in_layer,out_layer=net_5lstm(wv_model,args)
    
    model=Model(inputs=[in_layer],outputs=[out_layer])
    model.compile(optimizer=Adam(args.lr, beta_1=0.9, beta_2=0.9), loss='mse')
    
    plot_model(model, show_shapes=True, to_file=args.model_base+'.structure.png')
    with open(args.model_base+".chardict.json","w") as f:
        json.dump(char_dict,f)
    save_cb=ModelCheckpoint(filepath=args.model_base+".{epoch:02d}-{loss:.5f}-{val_loss:.5f}.hdf5",verbose=1,save_best_only=False)

    split=int(len(char_sequences)*0.8)
    train_ends=split
    val_begins=split
    vectors=wv_model.vectors
    #vectors=normalize(wv_model.vectors)
    while True:
        try:
            model.fit(char_sequences[:train_ends],vectors[:train_ends],batch_size=500,epochs=args.epochs,verbose=1, validation_data=(char_sequences[val_begins:],vectors[val_begins:]),callbacks=[save_cb])
            break
        except:
            traceback.print_exc()
            import pdb
            pdb.set_trace()
