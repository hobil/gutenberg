"""
Computes the text features.
"""
import numpy as np
import pandas as pd
import re
import nltk
import spacy
from collections import Counter
import time
import os
import sys
import pickle
from .metadata import load_metadata
from .lists import text_feature_labels, wrong_lemmatization

DEFAULT_TOKEN_PATTERN=r"(?u)\b\w+\b"
TOKEN_PATTERN = re.compile(DEFAULT_TOKEN_PATTERN)

STOPWORDS=nltk.corpus.stopwords.words('english')
SPACYDOC_PATH='../res/spacydocs'
FV_PATH='../res/featvec'
NLP = None

TOKENIZED_PATH='../res/tokenized'


def _compute_word_features(doc):
  """
  Computes avg word length and stop words proportion.
  """
  
  features=[]
  # Extract words consisting of alphanum symbols only
  words=TOKEN_PATTERN.findall(doc)
  
  # Compute avg word length
  avg_word_length=np.average([len(x) for x in words])
  features+=[avg_word_length]
  
  # Compute the proportion of stop words in the document
  stopwords=np.average([x.lower() in STOPWORDS for x in words])
  features+=[stopwords]
  return features

def _compute_sent_features(doc):
  """
  Compute avg sentence length and starting/ending symbols in the sentence.
  """
  
  features=[]
  # Create list of sentences form the document
  sents=nltk.sent_tokenize(doc)
  number_of_sentences=len(sents)
  words_in_sents=[nltk.word_tokenize(x) for x in sents]
  # Compute avg number of words in a sentence
  avg_words_in_sent=np.average([len(TOKEN_PATTERN.findall(sent)) for sent in sents])
  features+=[avg_words_in_sent]
  
  # Compute if the sentence ends with . , ? or !
  symbols=['.','!','?']
  last_sent_symbol=[sent[-1] if (sent[-1] in symbols) | (len(sent) < 2) else sent[-2] for sent in words_in_sents]
  last_sent_symbol_Counter=Counter(last_sent_symbol)
  last_sent_symbol_dot=1.0*last_sent_symbol_Counter['.']/number_of_sentences
  last_sent_symbol_question=1.0*last_sent_symbol_Counter['?']/number_of_sentences
  last_sent_symbol_exclamation=1.0*last_sent_symbol_Counter['!']/number_of_sentences
  features+=[last_sent_symbol_dot,last_sent_symbol_question,last_sent_symbol_exclamation]
  
  # Compute if the sentence starts or ends with quotation marks
  quotation_marks_set={"'","''",'"','""','“','‘','’','”'}
  quotation_beginning=np.average([sent[0] in quotation_marks_set for sent in sents])
  quotation_end=np.average([sent[-1] in quotation_marks_set for sent in sents])
  features+=[quotation_beginning,quotation_end]
  return features
  
def _compute_pos_features(d,book_id=None):
  """
  Computes pos features.
  d is tokenized text by spacy
  """

  # save tokens in format (lemma,pos)
  features=[]
  POS_SET=['ADJ','ADP','ADV','CONJ','DET','NOUN','NUM','PART','PRON','PROPN','PUNCT','VERB']
  
  # Compute POS distributions
  pos_Counter=Counter([x.pos_ for x in d])
  items_in_pos_set=max(1.0,sum([pos_Counter[x] for x in POS_SET]))
  
  for pos in POS_SET:
    features+=[(1.0*pos_Counter[pos] if pos in pos_Counter else 0)/items_in_pos_set]
    
  tag_Counter = Counter([x.tag_ for x in d])
  
  # Add VERB types
  VERB_TYPES=['MD','VB','VBD','VBG','VBN','VBP','VBZ']
  verb_freq=pos_Counter['VERB'] if 'VERB' in pos_Counter else 1.0
  for verb in VERB_TYPES:
    features+=[(1.0*tag_Counter[verb] if verb in tag_Counter else 0) /verb_freq]
  
  # Add PART types
  PART_TYPES=['TO','RP','POS']
  part_freq=pos_Counter['PART'] if 'PART' in pos_Counter else 1.0
  for part in PART_TYPES:
    features+=[(1.0*tag_Counter[part] if part in tag_Counter else 0) /part_freq]
      
  # Add NOUN types
  NOUN_TYPES=['NN','NNS','WP']
  noun_freq=pos_Counter['NOUN'] if 'NOUN' in pos_Counter else 1.0
  for noun in NOUN_TYPES:
    features+=[(1.0*tag_Counter[noun] if noun in tag_Counter else 0) /noun_freq]
    
  # Add ADV types
  ADV_TYPES=['RB','WRB','EX','RBS','RBR']
  adv_freq=pos_Counter['ADV'] if 'ADV' in pos_Counter else 1.0
  for adv in ADV_TYPES:
    features+=[(1.0*tag_Counter[adv] if adv in tag_Counter else 0) /adv_freq]
  
  # Add ADJ types
  ADJ_TYPES=['JJ','PRP$','WDT','JJR','JJS','PDT','WP$']
  adj_freq=pos_Counter['ADJ'] if 'ADJ' in pos_Counter else 1.0
  for adj in ADJ_TYPES:
    features+=[(1.0*tag_Counter[adj] if adj in tag_Counter else 0) /adj_freq]
  
  # Add PUNCT types
  PUNCT_TYPES=[',','.',':','HYPH']
  punct_freq=pos_Counter['PUNCT'] if 'PUNCT' in pos_Counter else 1.0
  for punct in PUNCT_TYPES:
    features+=[(1.0*tag_Counter[punct] if punct in tag_Counter else 0) /punct_freq]
  quotations=(tag_Counter["''"] if "''" in tag_Counter else 0) + (tag_Counter['``'] if '``' in tag_Counter else 0)
  brackets=(tag_Counter['-LRB-'] if '-LRB-' in tag_Counter else 0) + (tag_Counter['-RRB-'] if '-RRB-' in tag_Counter else 0)
  features+=[1.0*quotations / punct_freq, 1.0*brackets / punct_freq]

  return features



def compute_feature_vector(book_id,recompute=False):
  """
  Computes features for the given book if not already computed.
  Saves the text feature vector.
  """

  if not os.path.exists(FV_PATH):
    os.mkdir(FV_PATH)
  
  fv_path = os.path.join(FV_PATH,'%d.fv.pickle'%book_id)
  if os.path.exists(fv_path) & (not recompute):
    print(book_id,'book found in the db, loading, not computing')
    return pickle.load(open(fv_path,'rb'))
  
  from .load_book import load_and_filter_document
  doc = load_and_filter_document(book_id)
  if doc == None:
    print(book_id,'ERROR, book, not found, skipping')
    return
  
  features = []
  start = time.time()
  
  features += _compute_word_features(doc)
  features += _compute_sent_features(doc)

  global NLP
  d = NLP(doc,parse=False,entity=False)
  features += _compute_pos_features(d,book_id)
  print("%s : %f."%((str(book_id),time.time()-start)))
  
  pickle.dump(features,open(fv_path,'wb'))
  return features

def main(argv):
  """
  Computes text features for a given range of bok ids.
  """
  
  global NLP
  NLP=spacy.en.English(parse=False,entity=False)
  
  if len(argv) > 2:
    book_ids=range(int(argv[1]),int(argv[2]))
    df_meta = load_metadata()
    book_ids=[b for b in book_ids if b in df_meta.index]
  else:
    #book_ids=[1,11,12,30,74,76,82,161,1342,1661]
    book_ids=[22778]
  
  non_empty_book_ids=[]
  feature_vectors=[]
  
  start=time.time()
  
  for b in book_ids:
    fv=compute_feature_vector(b,recompute=True)
    if fv!=None:
      non_empty_book_ids+=[b]
      feature_vectors.append(fv)
  end=time.time()
  
  print("\nTotal time: %.2f"%(end-start))
  print("%.2f per book"%((end-start)/len(book_ids)))
  
  return pd.DataFrame(feature_vectors,index=non_empty_book_ids,columns = text_feature_labels).transpose()


if __name__=='__main__':
  main(sys.argv)

"""
def save_tuples(d,book_id):
  tokenized_file=os.path.join(TOKENIZED_PATH,str(book_id))
  tuples=[((x.lemma_ if x.lemma_ not in wrong_lemmatization else x.text),x.pos) for x in d]
  pickle.dump(tuples,open(tokenized_file,'wb'))
"""