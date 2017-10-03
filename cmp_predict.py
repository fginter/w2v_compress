import json
import six
assert six.PY3, "run me with python3"

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
from keras.models import load_model


from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge
from keras.layers.core import Masking, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback,ModelCheckpoint
import pickle
import sys
from keras.utils import plot_model
import keras.backend as K
from keras.utils import normalize


import cmp

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Sclassifier')
    parser.add_argument('--embeddings', help='embeddings file')
    parser.add_argument('--edim', type=int, default=200000, help='Embedding dimension')
    parser.add_argument('--max-seq-len', type=int, default=20, help='training data')
    parser.add_argument('--model', default='model-5lstm',help="Base name for the model file")
    parser.add_argument('--chardict',help="Read this chardict")
    parser.add_argument('--out', help="W2V .bin file name")
    parser.add_argument('--interactive', default=False, action="store_true", help="Interactive nearest")
    args = parser.parse_args()

    print("Loading w2v", args.embeddings,file=sys.stderr)
    wv_model=lwvlib.load(args.embeddings,args.edim,args.edim)
    print("Loading chardict", args.chardict, file=sys.stderr)
    with open(args.chardict) as f:
        char_dict=json.load(f)
    print("Loading model", args.model, file=sys.stderr)
    model=load_model(args.model)

    char_sequences=cmp.get_char_sequences(wv_model.words,char_dict,args.max_seq_len,learn_dict=False)
    predicted=model.predict(char_sequences,500) #500 is minibatch size
    new_w2v=lwvlib.WV(wv_model.words,predicted,None,[])
    if args.out:
        new_w2v.save_bin(args.out)
        for w in wv_model.words[120000:]:
            print(w)
    if args.interactive:
        while True:
            w=input("> ")
            char_sequences=cmp.get_char_sequences([w.strip()],char_dict,args.max_seq_len,learn_dict=False)
            predicted=normalize(model.predict(char_sequences))
            #print("predicted",predicted)
            nearest=new_w2v.nearest_to_normv(predicted[0],10)
            for sim,neighb in nearest:
                print("{:.2f} {:15s}".format(sim,neighb),"       ",end="")
                print()
            print()
            print()

    
