import numpy as np
# import marshal as pickle
import h5py
import sys
import os
# from tqdm import tqdm
# from ocr_model import OCR_model
import scipy.io as sio

fold = sys.argv[1]
fl = sys.argv[2]
epoch_n = sys.argv[3]

model_config = {
	'W1': 'dense_1_W',
	'b1': 'dense_1_b',
	'W2': 'dense_2_W', 
	'b2': 'dense_2_b', 
	'lstm_W1': 'lstmcell_1_W', 
	'lstm_U1': 'lstmcell_1_U', 
	'lstm_b1': 'lstmcell_1_b', 
	'lstm_W2': 'lstmcell_2_W', 
	'lstm_U2': 'lstmcell_2_U', 
	'lstm_b2': 'lstmcell_2_b', 
	'att_W1': 'attentiondecodercell_1_W1', 
	'att_W2': 'attentiondecodercell_1_W2', 
	'att_W3': 'attentiondecodercell_1_W3', 
	'att_W4': 'attentiondecodercell_1_W4', 
	'att_U': 'attentiondecodercell_1_U', 
	'att_b1': 'attentiondecodercell_1_b1', 
	'att_b2': 'attentiondecodercell_1_b2', 
	'att_b3': 'attentiondecodercell_1_b3', 
	'att_b4': 'attentiondecodercell_1_b4', 
}

def load_data(fn, data):
	file = h5py.File(fn, 'r')
	for k in data:
		data[k] = np.asarray(file[data[k]][:], dtype='float32')
	return data

def relu(x):
	return (x + np.abs(x))/2

def hard_sigmoid(x):
	return np.clip((x * 0.2) + 0.5, 0, 1)

def softmax(x, axis=1):
	e = np.exp(x-np.max(x, axis=axis, keepdims= True))
	return e / np.sum(e, axis=axis, keepdims= True)

def seq2seq_att(weights, x):
	def lstm_batch(idx, x, go_back=False):
		if go_back:
			x = x[:,::-1,:]
		x = np.matmul(x, weights['lstm_W%d'%idx]) + weights['lstm_b%d'%idx]
		h = np.zeros((x.shape[0], x.shape[1], 64), dtype='float32')
		c = np.zeros((x.shape[0], x.shape[1], 64), dtype='float32')
		for t in range(x.shape[1]):
			z = x[:,t,:] + np.matmul(h[:, max(t-1, 0), :], weights['lstm_U%d'%idx])
			z0 = z[:, :64]
			z1 = z[:, 64: 2 * 64]
			z2 = z[:, 2 * 64: 3 * 64]
			z3 = z[:, 3 * 64:]
			i = hard_sigmoid(z0)
			f = hard_sigmoid(z1)
			c[:, t,:] = f * c[:, max(t-1, 0),:] + i * np.tanh(z2)
			o = hard_sigmoid(z3)
			h[:, t,:] = o * np.tanh(c[:, t,:])
		return h

	def lstm(idx, x, go_back=False):
		if go_back:
			x = x[::-1,:]
		x = x.dot(weights['lstm_W%d'%idx]) + weights['lstm_b%d'%idx]
		h = np.zeros((x.shape[0], 64), dtype='float32')
		c = np.zeros((x.shape[0], 64), dtype='float32')
		for t in range(x.shape[0]):
			z = x[t,:] + h[max(t-1, 0),:].dot(weights['lstm_U%d'%idx])
			z0 = z[:64]
			z1 = z[64: 2 * 64]
			z2 = z[2 * 64: 3 * 64]
			z3 = z[3 * 64:]
			i = hard_sigmoid(z0)
			f = hard_sigmoid(z1)
			c[t,:] = f * c[max(t-1, 0),:] + i * np.tanh(z2)
			o = hard_sigmoid(z3)
			h[t,:] = o * np.tanh(c[t,:])
		return h


	def att_lstm_batch(H):
		energy_x = np.matmul(H, weights['att_W3'][:64,:]) + weights['att_b3']
		sio.savemat('extractfeature_mat/test_oov_%s_%s_energy_x_%s.mat'%(fold, fl, epoch_n), {'energy_x': energy_x})
		# print('energy_x:',type(energy_x), energy_x.shape)
		h = np.zeros((H.shape[0], 25, 64), dtype='float32')
		c = np.zeros((H.shape[0], 25, 64), dtype='float32')
		
		for t in range(25):
			# energy = energy_x + np.expand_dims(np.matmul(c[:, max(t-1, 0),:], weights['att_W3'][64:,:]), axis=1)
			# print('1 energy :',type(energy),energy.shape)
			# energy = relu(energy)
			# print('2 relu :',type(energy), energy.shape)
			# energy = np.matmul(energy, weights['att_W4']) + weights['att_b4']
			# print('3 matmul :',type(energy), energy.shape)
			# energy = softmax(energy)
			# print('4 softmax:',type(energy), energy.shape)
			# energy = np.transpose(energy, [0,2,1])
			# print('5 transpose :',type(energy), energy.shape)
			# x = np.matmul(energy, H)
			# print('6 matmul :',type(x), x.shape)
			# x = np.squeeze(x)
			# print('7 squeeze :',type(x), x.shape)
			# z = np.matmul(x, weights['att_W1']) + np.matmul(h[:,max(t-1, 0),:], weights['att_U']) + weights['att_b1']
			# print('z:',type(z), z.shape)
			energy = energy_x + np.expand_dims(np.matmul(c[:, max(t - 1, 0), :], weights['att_W3'][64:, :]), axis=1)
			energy = relu(energy)
			energy = np.matmul(energy, weights['att_W4']) + weights['att_b4']
			energy = softmax(energy)
			energy = np.transpose(energy, [0, 2, 1])
			x = np.matmul(energy, H)
			x = np.squeeze(x)
			z = np.matmul(x, weights['att_W1']) + np.matmul(h[:, max(t - 1, 0), :], weights['att_U']) + weights[
				'att_b1']
			z0 = z[:,:64]
			z1 = z[:,64: 2 * 64]
			z2 = z[:,2 * 64: 3 * 64]
			z3 = z[:,3 * 64:]
			i = hard_sigmoid(z0)
			f = hard_sigmoid(z1)
			c[:,t,:] = f * c[:,max(t-1, 0),:] + i * np.tanh(z2)
			o = hard_sigmoid(z3)
			h[:,t,:] = o * np.tanh(c[:,t,:])
		
		y = softmax(np.matmul(h, weights['att_W2']) + weights['att_b2'], axis=-1)
		return y


	def att_lstm(H):
		energy_x = H.dot(weights['att_W3'][:64,:]) + weights['att_b3']
		h = np.zeros((25, 64), dtype='float32')
		c = np.zeros((25, 64), dtype='float32')
		
		for t in range(25):
			energy = energy_x + c[max(t-1, 0),:].dot(weights['att_W3'][64:,:])
			energy = relu(energy)
			energy = energy.dot(weights['att_W4']) + weights['att_b4']
			energy = softmax(energy.T)
			energy = energy.reshape((1, 310))
			x = energy.dot(H)
			x = x.reshape((64,))
			z = x.dot(weights['att_W1']) + h[max(t-1, 0),:].dot(weights['att_U']) + weights['att_b1']
			z0 = z[:64]
			z1 = z[64: 2 * 64]
			z2 = z[2 * 64: 3 * 64]
			z3 = z[3 * 64:]
			i = hard_sigmoid(z0)
			f = hard_sigmoid(z1)
			f = hard_sigmoid(z1)
			c[t,:] = f * c[max(t-1, 0),:] + i * np.tanh(z2)
			o = hard_sigmoid(z3)
			h[t,:] = o * np.tanh(c[t,:])
		
		y = softmax(h.dot(weights['att_W2']) + weights['att_b2'])
		return y


	[sample, epoch, dim] = x.shape
	x = x.reshape((sample*epoch, dim))
	x = relu(x.dot(weights['W1']) + weights['b1'])
	h = relu(x.dot(weights['W2']) + weights['b2'])
	h = h.reshape((sample, epoch, 64))

	hl = lstm_batch(1, h, False)
	hr = lstm_batch(2, h, True)
	hr = hr[:,::-1,:]
	hlr = hl + hr

	out = att_lstm_batch(hlr)
	# print hl.shape
	# print hl[0,:,63]
	# print out.shape
	if not os.path.exists('extractfeature_mat'):
		os.mkdir('extractfeature_mat')
	sio.savemat('extractfeature_mat/test_oov_%s_%s_hl_%s.mat'%(fold, fl, epoch_n), {'hl': hl})
	sio.savemat('extractfeature_mat/test_oov_%s_%s_hr_%s.mat'%(fold, fl, epoch_n), {'hr': hr})
	sio.savemat('extractfeature_mat/test_oov_%s_%s_hlr_%s.mat'%(fold, fl, epoch_n), {'hlr': hlr})
	sio.savemat('extractfeature_mat/test_oov_%s_%s_out_%s.mat'%(fold, fl, epoch_n), {'out': out})

	# NO BATCH VERSION
	# out = np.zeros((sample, 30, 60))
	# for i in tqdm(range(sample), ascii=True):
	# 	hi = h[i,:,:]
	# 	hl = lstm(1, hi, False)
	# 	hr = lstm(2, hi, True)
	# 	hr = hr[::-1,:]
	# 	out[i, :, :] = att_lstm(hl + hr)

	return out


class MN_OCR_Model(object):
	# def __init__(self, weights_fn, tree_fn, glyph_map_fn, legacy_model_fn):
	def __init__(self, weights_fn):
		super(MN_OCR_Model, self).__init__()
		self.weights = load_data(weights_fn, model_config)

	def run(self, x):
		return seq2seq_att(self.weights, x)


test = MN_OCR_Model('model_oov_%s_%s/%s.h5'%(fold, fl, epoch_n))
test.run(np.load('./data/testing_set_oov_%s_%s_data.npy'%(fold, fl)))