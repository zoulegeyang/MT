from langconv import *
import sys
import re
import tensorflow as tf
 
print(sys.version)
print(sys.version_info)
 
# 转换繁体到简体
def cht_to_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line
 
# 转换简体到繁体
def chs_to_cht(line):
    line = Converter('zh-hant').convert(line)
    line.encode('utf-8')
    return line
	
def preprocessing_en_sentence(w):
    w = re.sub(r"([?.!])", r" \1 ", w)
    w = '<start> ' + w + ' <end>'
    return w

def preprocessing_ch_sentence(w):
    w = list(w.strip().replace(' ', ''))
    w = cht_to_chs(w)
    w = ' '.join(w)
    w = '<start> ' + w + ' <end>'
    return w
	
def create_dataset(path, num_samples):
    lines = open(path,encoding='utf-8').read().strip().split('\n')
    raw_word_pairs = [[w for w in line.split('\t')] for line in lines[: num_samples]]
    word_pairs = [[preprocessing_en_sentence(en),preprocessing_ch_sentence(ch)] for en, ch in raw_word_pairs]
    return word_pairs
	
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
        
        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
  
def max_length(tensor):
    return max(len(t) for t in tensor)
	
def load_dataset(path, num_examples):
    # creating cleaned input, output pairs
    pairs = create_dataset(path, num_examples)

    # index language using the class defined above    
    inp_lang = LanguageIndex(ch for en, ch in pairs)
    targ_lang = LanguageIndex(en for en, ch in pairs)
    
    # Vectorize the input and target languages
    
    # chinese sentences
    input_tensor = [[inp_lang.word2idx[s] for s in ch.split(' ')] for en, ch in pairs]
    
    # English sentences
    target_tensor = [[targ_lang.word2idx[s] for s in en.split(' ')] for en, ch in pairs]
    
    # Calculate max_length of input and output tensor
    # Here, we'll set those to the longest sentence in the dataset
    max_length_inp, max_length_tar = max_length(input_tensor), max_length(target_tensor)
    
    # Padding the input and output tensor to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=max_length_tar, 
                                                                  padding='post')
    
    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar