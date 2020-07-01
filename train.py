import os
import sys
import random
import numpy as np
from keras.models import Sequential
from seq2seq.models import AttentionSeq2Seq
from keras.layers import Dense, Activation, TimeDistributed
from keras.callbacks import CSVLogger
from keras.layers.recurrent import LSTM

from io_utils import save_data, load_data


#  THEANO_FLAGS=mode=FAST_RUN,device=cuda1,floatX=float32 python train.py fl2 200
def save_weights(model, pth):
    weights = {}
    for layer in model.layers:
        wn = layer.weights
        for w in wn:
            weights[str(w).replace('/', '_')] = w.get_value()
    save_data(pth, weights)


def load_weights(model, pth):
    for layer in model.layers:
        wn = layer.weights
        for w in wn:
            vn = str(w).replace('/', '_')
            wv = load_data(pth, vn)[vn]
            w.set_value(wv)


        

fl=sys.argv[1]
print fl
frame_length = int(fl[1])
epoch=int(sys.argv[2])

TIME_STEPS = int(round(308/frame_length)-1)  #319
NUM = 64
num_dim = 50*frame_length*2



model = Sequential()
model.add(TimeDistributed(Dense(64, activation='relu'), batch_input_shape=(None, TIME_STEPS, num_dim)))
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(AttentionSeq2Seq(input_dim=64, input_length=TIME_STEPS, hidden_dim=64, output_length=25, output_dim=50, depth=1))
model.add(TimeDistributed(Activation('softmax')))

model.compile(loss='sparse_categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
print(model.summary())

X = np.load('4folds_cross_validation/training_set_of_4cv_data_%s.npy'%fl)
Y = np.load('4folds_cross_validation/training_set_of_4cv_label.npy')

random.seed('alzhu_san')
idx = range(X.shape[0])
random.shuffle(idx)

tX = X[idx[:1000],:,:]
tY = Y[idx[:1000],:,:]
X = X[idx[1000:],:,:]
Y = Y[idx[1000:],:,:]

#load_weights(model, 'model_NR96/epoch_%d.h5'%331)
if not os.path.exists('model_%s'%fl):
    os.mkdir('model_%s'%fl)

def this_model():
    e = -1
    for i in range(100000):
        if os.path.exists('model_%s/epoch_%d.h5'%(fl,i)):
            e = i
        else:
            break
    if e >= 0:
        load_weights(model, 'model_%s/epoch_%d.h5'%(fl,e))


def next_model():
    for i in range(100000):
        if not os.path.exists('model_%s/epoch_%d.h5'%(fl,i)):
            return i

print(sys.argv)
this_model()

for i in range(epoch):
    print("Training the "+str(i+1)+" EPOCH")
    csv_logger = CSVLogger('log/train_%s.log'%fl, separator=',', append=True)
    model.fit(X, Y, validation_data=(tX, tY), nb_epoch = 1, batch_size=128, callbacks=[csv_logger])
    save_weights(model, 'model_%s/epoch_%d.h5'%(fl,next_model()))
