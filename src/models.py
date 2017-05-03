"""
Loads feature vectors with given parameters and splits them in train and test set.
"""
import os
import pickle
from scipy import stats
from scipy.sparse import csr_matrix
import pandas as pd
from metadata import load_metadata
import random
from collections import Counter
import numpy as np
import numbers


def load_fvs(df_type='bin', size='', norm=False, models_folder='../res/models', gram='bigram', combined = False):
  
  # text_features are already a df, just load it and normalize if needed
  if df_type=='text_feats':
    fvs_filename='text_features.pickle'
    fvs_filename_path=os.path.join(models_folder,fvs_filename)
    df=pickle.load(open(fvs_filename_path,'rb'))
    if norm:
      df=pd.DataFrame(stats.zscore(df.values),index=df.index,columns=df.columns)
    return df
  
  # rel, bin and tfidf are a triple of (sparse_matrix, bookids, vocabulary)
  if df_type=='rel':
    fvs_filename='bow_rel_'
  elif df_type=='tfidf':
    fvs_filename='bow_tfidf_'
  elif df_type == 'bin':
    fvs_filename='bow_bin_'
  else:
    print("False model type.")
    return
  
  fvs_filename += gram + size

  fvs_filename_path=os.path.join(models_folder,fvs_filename)+".pickle"
  fvs,bookids,words_index=pickle.load(open(fvs_filename_path,'rb'))
  
  if type(fvs) == csr_matrix:
    fvs=fvs.todense()
  if norm:
    fvs=fvs=stats.zscore(fvs)
  
  df = pd.DataFrame(fvs,index=bookids,columns=words_index)
  if combined == False:
    return df
  else:
    fvs_filename='text_features.pickle'
    fvs_filename_path=os.path.join(models_folder,fvs_filename)
    df_text=pickle.load(open(fvs_filename_path,'rb'))
    
    df[df_text.columns] = df_text
    return df

def split_into_train_and_test(df, attribute, train_ratio = 0.9, min_occurence = 1, exclude_set = None, sampling = None):
  
  if exclude_set is None:
    if attribute == 'author':
      exclude_set = set(['Various','Unknown','Anonymous'])
    else:
      exclude_set = set()
      
  # filter not wanted bookids
  df_meta = load_metadata()
  df_meta = df_meta[df_meta.ix[:, attribute].notnull()]
  c = Counter(df_meta.ix[:, attribute]).most_common()
  c = [x for x in c if x[1] >= min_occurence]
  # values available for the label after filtering
  attribute_values = set([x[0] for x in c]) - set(exclude_set)
  # bookids with feasible label parameters
  idxs = df_meta[df_meta.ix[:, attribute].apply(lambda x: x is not None and x in attribute_values)].index
  
  bookids=[b for b in df.index if b in idxs]

  if sampling is None:
    print("splitting quick")
    #permutation=list(range(len(bookids)))
    #random.shuffle(permutation,lambda : SEED)
    #random.shuffle(permutation)
    
    # shuffled bookids
    #bookids_shuffl=[bookids[i] for i in permutation]
    
    
    # adding reserved books to the end to come into test set
    ids_avail_for_train = [b for b in bookids if b not in RESERVED_FOR_TEST]
    random.shuffle(ids_avail_for_train)
    reserved_ids = [b for b in RESERVED_FOR_TEST if b in bookids]
    bookids_shuffled = ids_avail_for_train + reserved_ids
    
    # divide bookids into train and test set
    border_idx=int(len(bookids_shuffled)*train_ratio)
    bookids_train=bookids_shuffled[:border_idx]
    bookids_test=bookids_shuffled[border_idx:]
  
  elif sampling == 'undersample':
    print("undersampling")
    available_ids = [i for i in bookids if i not in RESERVED_FOR_TEST]
    attribute_counter = Counter(df_meta.ix[available_ids, attribute])
    minimum_occurence = min(list(attribute_counter.values()))
    train_size_per_class = int(minimum_occurence * train_ratio)
    
    bookids_train = []
    for a in attribute_counter:
      col = df_meta.ix[available_ids,attribute]
      bookids_train += df_meta.ix[available_ids].query('@col == @a').sample(train_size_per_class).index.tolist()
    
    bookids_test = [b for b in bookids if b not in bookids_train]
    
    random.shuffle(bookids_train)
    random.shuffle(bookids_test)
  
  elif sampling == 'oversample':
    print("oversampling")
    available_ids = [i for i in bookids if i not in RESERVED_FOR_TEST]
    attribute_counter = Counter(df_meta.ix[available_ids, attribute])
    maximum_occurence = max(list(attribute_counter.values()))
    train_size_per_class = int(maximum_occurence * train_ratio)
    
    bookids_train = []
    for a in attribute_counter:
      col = df_meta.ix[available_ids,attribute]
      ids = df_meta.ix[available_ids].query('@col == @a').index.tolist()      
      random.shuffle(ids)
      ids_allowed_in_train = int(len(ids)*train_ratio)
      
      repeat_n = int(np.ceil(train_size_per_class / ids_allowed_in_train))
      ids2 = np.tile(ids[:ids_allowed_in_train],repeat_n)
      bookids_train += ids2[:train_size_per_class].tolist()
    
    bookids_train_set = set(bookids_train)
    bookids_test = [b for b in bookids if b not in bookids_train_set]
    random.shuffle(bookids_train)
    random.shuffle(bookids_test)
    
  
  print("training set size: %d"%len(bookids_train))
  print("test set size: %d"%len(bookids_test))
  
  X_train=df.ix[bookids_train].values
  X_test=df.ix[bookids_test].values
  
  y_train=df_meta.ix[bookids_train,attribute]
  y_test=df_meta.ix[bookids_test,attribute]
  
  return X_train, y_train, X_test, y_test

def evaluate(y_train,y_test,y_pred):
  evaluation=dict()
  modus=Counter(y_train).most_common()[0][0]
  y_naive=np.repeat(modus,len(y_test))
  print("\n===================")
  correct=sum(y_pred==y_test)/len(y_test)
  evaluation['correct']=correct
  correct_naive=sum(y_test==y_naive)/len(y_test)
  evaluation['correct_naive']=correct_naive
  
  print("%.3f : correct"%correct)
  print("%.3f : correct naive"%correct_naive)
  
  if isinstance(y_pred[0], numbers.Number):
    class_dist=sum(np.abs(y_pred-y_test))/len(y_test)
    evaluation['class_dist'] = class_dist
    naive_class_dist=sum(np.abs(y_naive-y_test))/len(y_test)
    evaluation['naive_class_dist'] = naive_class_dist
    
    print("%.3f : avg prediction distance from actual class"%class_dist)
    print("%.3f : avg naive distance from actual class"%naive_class_dist)
  
  
  print("===================\n")
  return evaluation

RESERVED_FOR_TEST = {1, # The Declaration of Independence of the United States of America by Thomas Jefferson
                     10, # The King James Version of the Bible
                     11, # Alice's Adventures in Wonderland by Lewis Carroll
                     2591, # Grimms' Fairy Tales by Jacob Grimm and Wilhelm Grimm
                     74, # The Adventures of Tom Sawyer by Mark Twain
                     82, # Ivanhoe: A Romance by Walter Scott
                     98, # A Tale of Two Cities by Charles Dickens
                     174, # The Picture of Dorian Gray by Pscar Wilde
                     236, # The Jungle Book by Rudyard Kipling
                     345, # Dracula by Bram Stoker
                     829, # Gulliver's Travels into Several Remote Nations of the World by Jonathan Swift
                     910, # White Fang by Jack London
                     996, # Don Quixote by Cervantes
                     1497, # Republic by Platon
                     1342, # Pride and Prejudice by Jane Austen
                     1661, # The Adventures of Sherlock Holmes by Arthur Conan Doyle
                     2383, # The Canterbury Tales by Geoffrey Chaucer
                     2434, # New Atlantis by Francis Bacon
                     2600, # War and Peace by graf Leo Tolstoy
                     2701, # Moby Dick; Or, The Whale by Herman Melville
                     3207, # Leviathan by Thomas Hobbes
                     16328, # Beowulf by J. Lesslie Hall
                     36, # War of the Worlds
                     84, # Frankenstein
                     }
