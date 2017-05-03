"""
Creates a model from precomputed word counts.
"""

import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from bows import get_count_df
from scipy.sparse import csr_matrix

def main(args):
  
  gram = ''
  bow_type = ''
  bow_size = None
  bow_size_str = ''
  bow_folder = '../res/models'
  
  if 'u' in args:
    gram = 'unigram'
  
  if 'b' in args:
    gram = 'bigram'
  
  if 'bin' in args:
    bow_type = 'bin'
  
  if 'rel' in args:
    bow_type = 'rel'
  
  if 'tfidf' in args:
    bow_type = 'tfidf'
  
  if '2k' in args:
    bow_size = 2000
    bow_size_str = '2k'
  
  if '1k' in args:
    bow_size = 1000
    bow_size_str = '1k'
  
  if '10k' in args:
    bow_size = 10000
    bow_size_str = '10k'
  
  if '15k' in args:
    bow_size = 15000
    bow_size_str = '15k'
  
  if '500' in args:
    bow_size = 500
    bow_size_str = '500'
  
  if '100' in args:
    bow_size = 100
    bow_size_str = '100'
  
  if '10' in args:
    bow_size = 10
    bow_size_str = '10'
  
  if '20' in args:
    bow_size = 10
    bow_size_str = '20'
  
  if '50' in args:
    bow_size = 50
    bow_size_str = '50'
  
  if '20k' in args:
    bow_size = 20000
    bow_size_str = '20k'
    
  if '30k' in args:
    bow_size = 30000
    bow_size_str = '30k'
    
  if '5k' in args:
    bow_size = 5000
    bow_size_str = '5k'
  
  if '8k' in args:
    bow_size = 8000
    bow_size_str = '8k'
  
  if bow_type not in ['bin', 'rel', 'tfidf']:
    print("Choose type between 'bin', 'rel' and 'tfidf'.")
    return
  
  if gram == '':
    print("Choose between unigram and bigram.")
    return
  
  
  bow_filename = 'bbw_'+gram+bow_size_str
  bow_file_path = os.path.join(bow_folder, bow_filename) + '.pickle'
  
  if os.path.exists(bow_file_path):
    bow,bookids,words_index=pickle.load(open(bow_file_path,'rb'))
    print("bow loaded")
  else:
    bow,bookids,words_index=get_count_df(bigrams=gram=='bigram', max_features = bow_size)
    pickle.dump((bow,bookids,words_index),open(bow_file_path,'wb'))
    print("bow saved")
  
  if bow_type == 'bin':
    bow = bow > 0
    
  elif bow_type == 'rel':
    bow = bow.todense()
    fvs = np.zeros(bow.shape)
    for n,b in enumerate(bow):
      bsum = b.sum()
      if bsum == 0:
        bsum = 1
      fvs[n] = b / bsum
    bow = csr_matrix(fvs)
    
  elif bow_type == 'tfidf':
    transformer = TfidfTransformer()
    bow = transformer.fit_transform(bow)
    bow = csr_matrix(bow, dtype = np.float32)
    
    transformer_filename = 'transformer_' + bow_type + '_' + gram + bow_size_str
    transformer_file_path = os.path.join(bow_folder, transformer_filename) + '.pickle'
    pickle.dump(transformer, open(transformer_file_path, 'wb'))
    vocab_filename = 'vocab_' + bow_type + '_' + gram + bow_size_str
    vocab_file_path = os.path.join(bow_folder, vocab_filename) + '.pickle'
    pickle.dump(words_index, open(vocab_file_path, 'wb'))
  
  fvs_filename = 'bow_' + bow_type + '_' + gram + bow_size_str
  
  fvs_file_path = os.path.join(bow_folder, fvs_filename) + '.pickle'
  
  pickle.dump((bow, bookids, words_index), open(fvs_file_path, 'wb'))
  print("%s saved" % fvs_filename)

if __name__ == '__main__':
  main(sys.argv[1:])
