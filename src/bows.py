"""
Selects a subset of features from the computed word counts.
"""
import pickle
import numpy as np
import pandas as pd
import os
import re
from stop_words import stop_words
from metadata import load_metadata
 
MODELS_FOLDER='../res/models'
n=42447

BOW_FILENAME={False: os.path.join(MODELS_FOLDER,'matrix_1_%d'%n), True: os.path.join(MODELS_FOLDER,'matrix_2_%d'%n)}
BOOKIDS_FILENAME={False: os.path.join(MODELS_FOLDER,'bookids_1_%d'%n), True: os.path.join(MODELS_FOLDER,'bookids_2_%d'%n)}
VOCABS_FILENAME={False: os.path.join(MODELS_FOLDER,'vocab_1_%d'%n), True:os.path.join(MODELS_FOLDER,'vocab_2_%d'%n)}
df_meta = load_metadata()

bigrams=True
min_df=0.0
max_df=1.0
max_features=None

def _filter_out_numbers(words,filter_out_numbers):
  if filter_out_numbers:
    NUMBER_REGEXP=r"^[0-9]+$"
    NUMBER_PATTERN = re.compile(NUMBER_REGEXP)
    non_number_words=[w for w in words.index if not NUMBER_PATTERN.match(w)]
    return words.ix[non_number_words]
  else:
    return words
  
def _filter_out_mindf_maxdf(words,min_df,max_df,no_of_docs):
  min_abs=min_df * no_of_docs
  max_abs=max_df * no_of_docs
  return words[words.counts.apply(lambda x: x >= min_abs  and x <= max_abs )]

def _select_max_features(words,max_features):
  if max_features==None:
    return words
  else:
    return words.sort_values(['sums','counts'], ascending=[False,False]).head(max_features).sort_values(['id'])

def get_count_df(selected_bookids=None, max_features=None, min_df=0.0, max_df=1.0, bigrams=False, filter_out_numbers=True, stopwords=stop_words):
  
  bow=pickle.load(open(BOW_FILENAME[bigrams],'rb'))
  print("bow loaded")
  bookids=np.array(pickle.load(open(BOOKIDS_FILENAME[bigrams],'rb')))
  # check if selected_bookids is a subset of bookids
  if selected_bookids != None:
    not_found_selected_bookids=[b for b in selected_bookids if b not in bookids]
    if len(not_found_selected_bookids) > 0:
      print("ERROR, following bookids not found: ",not_found_selected_bookids)
      return
  if selected_bookids == None:
    bookids_bool=[b in df_meta.index for b in bookids]
  else:
    bookids_bool=[(b in df_meta.index) and (b in selected_bookids) for b in bookids]

  bookids=bookids[bookids_bool]
  idxs=[i for i,b in enumerate(bookids_bool) if b]
  bow=bow[idxs]
  no_of_docs=len(bookids)
  vocab=pickle.load(open(VOCABS_FILENAME[bigrams],'rb'))
  vocab_keys=sorted(vocab, key=vocab.get)
  # create a df with word sums and counts in the corpus
  print("creating sums")
  word_sums=np.sum(bow,axis=0).A1
  print("creating counts")
  word_no_of_docs=np.sum(bow>0,axis=0).A1
  print("creating words")
  words=pd.DataFrame([range(len(vocab)),word_sums,word_no_of_docs],columns=vocab_keys,index=['id','sums','counts']).T
  print("words creating")
  # filter out stopwords
  #words=words.ix[[w for w in words.index if w not in stop_words]]
  words=words.ix[[w for w in words.index if any([x not in stop_words for x in w.split()])]]
  #print("maximum: %d"%np.max(bow))
  # filter out numbers if requested
  words=_filter_out_numbers(words,filter_out_numbers)
  # filter out words based on document frequency
  words=_filter_out_mindf_maxdf(words,min_df,max_df,no_of_docs)
  # select top n features based on total occurence in corpus
  words=_select_max_features(words, max_features)
  
  indices=words.id
  print("returning")
  return bow[:,indices],bookids,words.index

