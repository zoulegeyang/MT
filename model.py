from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
# 可以不用sess.run()就可以运行tensor
tf.enable_eager_execution()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model_util import gru, Encoder, Decoder,loss_function,plot_attention
from data_util import load_dataset,preprocessing_ch_sentence

import unicodedata
import numpy as np
import os
import time


class Model():

	def __init__(self):
		self.name = "machine translation"
		self.BUFFER_SIZE  = None
		
		 
		
	
	def load_data(self, path='./dataset/mt_en2ch.txt', num_examples = 20000, batch_size=64):
		print(">>>>>正在加载训练数据")
		
		input_tensor, target_tensor, self.inp_lang, self.targ_lang, self.max_length_inp, self.max_length_targ = load_dataset(path, num_examples)
		
		# Creating training and validation sets using an 80-20 split
		self.input_tensor_train, self.input_tensor_val, self.target_tensor_train, self.target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
		self.vocab_inp_size = len(self.inp_lang.word2idx)
		self.vocab_tar_size = len(self.targ_lang.word2idx)
		self.BUFFER_SIZE = len(self.input_tensor_train)
		self.dataset = tf.data.Dataset.from_tensor_slices((self.input_tensor_train, self.target_tensor_train)).shuffle(self.BUFFER_SIZE).batch(batch_size, drop_remainder=True)
		
		print(">>>>>加载完毕")
		
	def prepare_model(self,units=1024, batch_size=64, embedding_dim=256):
		self.BATCH_SIZE = batch_size
		self.N_BATCH = self.BUFFER_SIZE//batch_size
		self.embedding_dim = embedding_dim
		self.units = units
		self.encoder = Encoder(self.vocab_inp_size, self.embedding_dim, self.units, self.BATCH_SIZE)
		self.decoder = Decoder(self.vocab_tar_size, self.embedding_dim, self.units, self.BATCH_SIZE)
		
		
	def load_model(self,checkpoint_dir = './training_checkpoints' ):
		self.prepare_model()
		optimizer = tf.train.AdamOptimizer()

		checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
		checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=self.encoder,
                                 decoder=self.decoder)
		#如果存在已经保存的模型，则加载已经训练的模型参数
		if tf.train.latest_checkpoint(checkpoint_dir) :
			checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
		return checkpoint,optimizer,checkpoint_prefix
		
	def train(self, EPOCHS = 10):
		checkpoint,optimizer,checkpoint_prefix = self.load_model()
		print(">>>>>>开始训练")
		for epoch in range(EPOCHS):
			start = time.time()
			
			hidden = self.encoder.initialize_hidden_state()
			total_loss = 0
			
			for (batch, (inp, targ)) in enumerate(self.dataset):
				loss = 0
				
				with tf.GradientTape() as tape:
					enc_output, enc_hidden = self.encoder(inp, hidden)
					dec_hidden = enc_hidden
					
					dec_input = tf.expand_dims([self.targ_lang.word2idx['<start>']] * self.BATCH_SIZE, 1)
					
					#teacher forcing - feeding the target as the next input
					for t in range(1, targ.shape[1]):
						# passing enc_output to the decoder
						predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
						
						loss += loss_function(targ[:, t], predictions)
						
						# using teacher forcing
						dec_input = tf.expand_dims(targ[:, t], 1)
						
				batch_loss = (loss / int(targ.shape[1]))
				
				total_loss += batch_loss
				
				variables = self.encoder.variables + self.decoder.variables
				
				gradients = tape.gradient(loss, variables)
				
				optimizer.apply_gradients(zip(gradients, variables))
				
				if batch % 100 == 0:
					print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
																 batch,
																 batch_loss.numpy()))
			# saving (checkpoint) the model every 2 epochs
			if (epoch + 1) % 2 == 0:
				checkpoint.save(file_prefix = checkpoint_prefix)
			
			print('Epoch {} Loss {:.4f}'.format(epoch + 1,
												total_loss / self.N_BATCH))
			print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
		
		
	def translate(self,sentence):
		checkpoint, _ ,_= self.load_model()
		attention_plot = np.zeros((self.max_length_targ, self.max_length_inp))
		
		sentence = preprocessing_ch_sentence(sentence)
		
		inputs = [self.inp_lang.word2idx[i] for i in sentence.split(' ')]
		inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=self.max_length_inp, padding='post')
		inputs = tf.convert_to_tensor(inputs)
		
		result = ''
		
		hidden = [tf.zeros((1, self.units))]
		enc_out, enc_hidden = self.encoder(inputs, hidden)
		
		dec_hidden = enc_hidden
		dec_input = tf.expand_dims([self.targ_lang.word2idx['<start>']], 0)
		
		for t in range(self.max_length_targ):
			predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)
			
			#storing the attention weights to plot later
			attention_weights = tf.reshape(attention_weights, (-1,))
			attention_plot[t] = attention_weights.numpy()
			
			predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
			
			result += self.targ_lang.idx2word[predicted_id] + ' '
			
			if self.targ_lang.idx2word[predicted_id] == '<end>':
				print('Input: {}'.format(sentence))
				print('Predicted translation: {}'.format(result))
    
				#attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
				#plot_attention(attention_plot, sentence.split(' '), result.split(' '))
				break
			
			#the predicted ID is fed back into the model
			dec_input = tf.expand_dims([predicted_id], 0)
			
			
        
		


if __name__ == "__main__":
	model = Model()
	model.load_data()
	#model.train()
	sentence = input("请输入要翻译的中文句子：")
	model.translate(sentence)
	

