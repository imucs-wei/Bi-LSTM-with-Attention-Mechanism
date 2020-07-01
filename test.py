import os
import sys
import random
import numpy as np
import marshal as pickle
from tqdm import tqdm
from keras.models import Sequential
from seq2seq.models import AttentionSeq2Seq
from keras.layers import Dense, Activation, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.callbacks import CSVLogger

from io_utils import save_data, load_data

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


#  THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python test.py fold1 fl1 1


fl=sys.argv[1]
print fl
frame_length = int(fl[1])
TIME_STEPS = int(round(308/frame_length)-1)  #308 is the image's max height
NUM = 64
num_dim = 50*frame_length*2


model = Sequential()
model.add(TimeDistributed(Dense(64, activation='relu'), batch_input_shape=(None, TIME_STEPS, num_dim)))
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(AttentionSeq2Seq(input_dim=64, input_length=TIME_STEPS, hidden_dim=64, output_length=25, output_dim=50, depth=1))
model.add(TimeDistributed(Activation('softmax')))
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

print sys.argv[2]
load_weights(model, 'model_%s/epoch_%s.h5'%(fl,sys.argv[2]))
text_lst = "$abcdefghijklmnopqrstuvwxyzABCDEGHIJKLMNOPQRSTUVZ#"


with open('tree', 'rb') as f:
	tree = pickle.load(f)

# find the next possible character 
def search_key(string):
	cur = tree
	string = '$'+string
	if string[-1] == '#': # short cut for speed
		return ['#']
	for x in string:
		cur = cur[x]
	return list(set(cur.keys()) & set(text_lst))

# return the top-N candidates
def top(k, n):
	sorted_x = sorted(k.items(), key=lambda x:x[1], reverse=True)
	return {x[0]:x[1] for x in sorted_x[:n]}

# remove space holder
def normal_word(w):
	return w.replace('$', '').replace('#', '')

# viterbi decoder 
def viterbi(text, N=4):
	keys = search_key('')
	res = {k: text[0, text_lst.index(k)] for k in keys}
	res = top(res, n=N)
	for i in range(1, text.shape[0]):
		tmp = {}
		for k,v in res.items():
			keys = search_key(k)
			for t in keys:
				tmp[k+t] =  v + text[i, text_lst.index(t)]
		res = top(tmp, n=N)		
	res = {normal_word(k): res[k] for k in res}
	return res



def recognize_text(x, beam=10, top_n=10):
	# x[:,:,text_lst.index('#')] = x[:,:,text_lst.index('#'):].sum(axis=-1)
	d = np.log(x)
	ret = ['']*d.shape[0]
	for i in tqdm(range(d.shape[0])):
		ret[i] = top(viterbi(d[i,:,:], beam), top_n)
	return ret


def top_acc(res, ref):
	rights = np.zeros((len(res[0]),))
	for i in tqdm(range(len(res))):
		k = res[i]
		sorted_x = sorted(k.items(), key=lambda x:x[1], reverse=True)
		r = ref[i]
		for j in range(len(sorted_x)):
			if sorted_x[j][0].strip() == r.strip():
				rights[j:] += 1

	return rights / len(res)

if 1:
	# testing_set_fold1_of_4cv_data_fl1.npy  testing_set_fold1_of_4cv_label.npy
	X = np.load('4folds_cross_validation/testing_set_of_4cv_data_%s.npy'%(fl))
	Y = np.load('4folds_cross_validation/testing_set_of_4cv_label.npy')
	print X.shape

	e = model.predict(X, batch_size=128, verbose=1)
	t = recognize_text(e)
    #testing_set_fold1_of_4cv_label.lst
	with open('4folds_cross_validation/testing_set_of_4cv_label.lst') as f:
		ref = f.readlines()

	# print top_acc(t, ref)
	result = top_acc(t, ref)
	with open('log/test_%s_results.log'%(fl) , 'a') as tt:
		tt.write('test acc with model %s \n' % sys.argv[2])
		tt.write(str(result).replace("\n", "") + "\n\n")
	tt.close()
	print result

else:
	e = np.load('E.npy')
	t = recognize_text(e[:2,:,:])
	print t

	with open('text.txt.ITMD') as f:
		ref = f.readlines()

	print top_acc(t, ref)
