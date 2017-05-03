"""
Predicts author, epoch, category and genres for the given text.
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import spacy
from spacy.symbols import *
from src.text_features import _compute_pos_features,_compute_sent_features,_compute_word_features
from src.lists import stop_words, wrong_lemmatization, text_feature_labels

models_folder = 'models/'
subjects = ['adventure_stories', 'love_stories', 'short_stories', 'historical_fiction', 'science_fiction', 'poetry', 'drama', 'detective_and_mystery_stories']

print("loading spacy pos tagger")
nlp=spacy.en.English(parse=False,entity=False)
print("spacy pos tagger loaded")


def load_vocab(model_type = 'tfidf', size = '2k'):
  """
  Loads a listed of top $n$ words in the vocabulary.
  """
  file_path = os.path.join(models_folder, 'vocab_%s_bigram%s.pickle'%(model_type,size))
  return pickle.load(open(file_path,'rb'))

def load_transformer(model_type = 'tfidf', size = '2k'):
  """
  Loads the tfidf transformer for the given number of words in vocabulary.
  """
  file_path = os.path.join(models_folder, 'transformer_%s_bigram%s.pickle'%(model_type,size))
  return pickle.load(open(file_path,'rb'))

def compute_text_features(doc_text,tagged_doc):
  """
  Computes text features.
  """
  features=[]
  features+=_compute_word_features(doc_text)
  features+=_compute_sent_features(doc_text)
  features+=_compute_pos_features(tagged_doc)
  # return as data frame with correct imported labels
  return pd.DataFrame([features], columns = text_feature_labels, index = ['sample'])

def compute_bow_features(d):
  """
  Compute feature vector relating to the individual word frequencies.
  """
  # lemmatize
  tuples=[((x.lemma_ if x.lemma_ not in wrong_lemmatization else x.text),x.pos) for x in d]
  # create unigrams
  tokens1=[x[0] for x in tuples if x[0] not in stop_words]
  # create bigrams
  tokens2=[x[0] +' '+tuples[n+1][0] for n,x in enumerate(tuples[:-1])]
  # filter bigrams containing stop words
  tokens2 = [t for t in tokens2 if all([x not in stop_words for x in t.split()])]
  # count terms
  c12 = Counter(tokens1+tokens2)
  # load vocabulary and tfidf transformer
  tfidf2k_vocab = load_vocab(model_type = 'tfidf', size = '2k')
  tfidf2k_transf = load_transformer(model_type = 'tfidf', size = '2k')
  tfidf2k_vocab_set = set(tfidf2k_vocab)
  # convert the feature vector to data frame
  df_tfidf2k = pd.DataFrame(np.zeros([1,2000]),index = ['sample'], columns = tfidf2k_vocab)
  for k,v in c12.items():
    if k in tfidf2k_vocab_set:
      df_tfidf2k[k] = v
  df_tfidf2k_transformed = pd.DataFrame(tfidf2k_transf.transform(df_tfidf2k).todense(), index = ['sample'], columns = tfidf2k_vocab)
  
  df_binary = df_tfidf2k > 0
  
  return df_tfidf2k_transformed, df_binary

def compute_features(doc):
  """
  Computes feature vectors needed for all the predictions.
  """
  print("computing POS tags")
  d = nlp(doc,parse=False,entity=False)
  print("computing text features")
  text_features = compute_text_features(doc,d)
  print("computing bow features")
  df_tfidf2k_transformed,df_binary = compute_bow_features(d)
  features = dict()
  features['binary'] = df_binary
  features['tfidf'] = df_tfidf2k_transformed
  df_tfidf2k_c = df_tfidf2k_transformed.copy()
  df_tfidf2k_c[text_features.columns] = text_features
  features['combined'] = df_tfidf2k_c
  return features

def load_model(attribute):
  """
  Loads model for the given attribute to be predicted.
  """
  
  model_filename = os.path.join(models_folder, attribute + '_model.pickle')
  if os.path.exists(model_filename):
    return pickle.load(open(model_filename,'rb'))
  else:
    print('%s model does not exist.' % attribute)
    return
  
def make_prediction(all_features,attribute,min_prob=0.01):
  """
  Returns the prediction result with its probability score.
  """
  if attribute == 'author':
    features = np.atleast_2d(all_features['binary'])
  else:
    features = np.atleast_2d(all_features['combined'])
  model = load_model(attribute)
  predictions = model.predict(features,min_prob=min_prob)
  return predictions[0]

def print_prediction(attribute,prediction):
  """
  Prints the result for multiclass attribute.
  """
  if attribute == 'epoch_names':
    a = 'epoch'
  elif attribute == 'lcc_class_simple':
    a = 'category'
  else:
    a = attribute
  
  print("\n======%s======"%a.upper())
  for k,score in prediction:
    print("%.3f : "%score, k)
  print("======================")
  
def print_usage():
  """
  Prints usage if user entered parameters in a wrong form.
  """
  
  print("Wrong parameter, usage:")
  print("python3 inspect_text.py [document_text]")
  print("OR")
  print("python3 inspect_text.py -f [filename]")
  return

def main(args):
  """ Accepts 1 string containing the text of the book.
  Or 2 strings where the first one is "-f" and the second one states the location of the book text.
  """
  
  if len(args) == 0:
    print_usage()
    return
  elif len(args) == 1:
    if type(args[0]) != str:
      print_usage()
      return
    else:
      doc = args[0]
  elif len(args) == 2:
    if args[0] != '-f':
      print_usage()
      return
    elif not os.path.exists(args[1]):
      print("File %s does not exist." % args[1])
      return
    else:
      doc = open(args[1]).read()
  else:
    print_usage()
  
  features = compute_features(doc)
  
  author_prediction = make_prediction(features,'author')
  print_prediction('author',author_prediction)
  epoch_predictions = make_prediction(features,'epoch_names')
  print_prediction('epoch_names',epoch_predictions)
  category_predictions = make_prediction(features,'lcc_class_simple')
  print_prediction('lcc_class_simple',category_predictions)
  
  subject_dict = dict()
  for s in subjects:
    subject_dict[s] = make_prediction(features,s)
  subjects_sorted = sorted(subject_dict.items(), key = lambda x: x[1], reverse = True)
  print("\n\n======SUBJECTS====")
  for k,v in subjects_sorted:
    print("%.3f : "%v, k)
  print("====================")

if __name__ == '__main__':
  main(sys.argv[1:])
