"""
Computes the word counts for all documents with ids within the specified range.
Performs tokenization, lemmatization and creates bigrams.
"""
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from spacy.symbols import *
from load_book import load_and_filter_document
import time
import numpy as np
import pickle
import sys
import os
import re
from scipy.sparse import csr_matrix

DEFAULT_TOKEN_PATTERN=r"'?\b\w+\b$"
TOKEN_PATTERN = re.compile(DEFAULT_TOKEN_PATTERN)
TOKENIZED_PATH='../res/tokenized'
MODELS_FOLDER='../res/models3'
BOOKIDS=[]
nlp = spacy.en.English(parse=False,entity=False)

wrong_lemmatization={"'","-PRON-","be","say","have"}


def preprocess(b_doc):
  return b_doc

def is_token_ok(t,pos={},lemmas={}):
  return(t.pos_ not in pos) & (t.lemma_ not in lemmas)
  
def is_tuple_ok(tup,pos={},lemmas={}):
  
  return(tup[1] not in pos) & (tup[0] not in lemmas)

def tokenize1(b_doc):
  b,doc=b_doc
  tokenized_file=os.path.join(TOKENIZED_PATH,str(b))  
  if os.path.exists(tokenized_file):
    tuples=pickle.load(open(tokenized_file,'rb'))
    print('loading   ',b)
  else:
    print('tokenizing',b)
    global nlp
    d=nlp(doc,parse=False,entity=False)
    # save model
    tuples=[((x.lemma_ if x.lemma_ not in wrong_lemmatization else x.text.lower()),x.pos) for x in d]
    pickle.dump(tuples,open(tokenized_file,'wb'))
  tokens1=[x[0].lower() for x in tuples if TOKEN_PATTERN.match(x[0])]
  return tokens1

def tokenize2(b_doc):
  b,doc=b_doc
  tokenized_file=os.path.join(TOKENIZED_PATH,str(b))  
  if os.path.exists(tokenized_file):
    tuples=pickle.load(open(tokenized_file,'rb'))
    print('loading   ',b)
  else:
    print('tokenizing',b)
    global nlp
    d=nlp(doc,parse=False,entity=False)
    # save model
    tuples=[((x.lemma_ if x.lemma_ not in wrong_lemmatization else x.text.lower()),x.pos) for x in d]
    pickle.dump(tuples,open(tokenized_file,'wb'))
  tokens1=[x[0].lower() for x in tuples if TOKEN_PATTERN.match(x[0])]
  tokens2=[x[0].lower() +' '+tuples[n+1][0].lower() for n,x in enumerate(tuples[:-1]) if TOKEN_PATTERN.match(x[0]) and TOKEN_PATTERN.match(tuples[n+1][0])]
  return tokens1+tokens2

def get_docs(bookids,all_exist=True,chapter_length=None):
  global BOOKIDS
  BOOKIDS = []
  
  for b in bookids:
    tokenized_file=os.path.join(TOKENIZED_PATH,str(b))
    if all_exist & os.path.exists(tokenized_file):
        if chapter_length==None:
          BOOKIDS+=[b]
          yield b,''
        else:
          tuples=pickle.load(open(tokenized_file,'rb'))
    else:
    
      doc=load_and_filter_document(b)
      if doc == None:
        print('continue')
        continue
      print('adding',b)
      BOOKIDS+=[b]
      print('yielding',b,doc[:10])
      yield b,doc


def get_word_freq(matrix,vocab,words):
  result=[]
  n=matrix.shape[0]
  for w in words:
    result+=[(w,str(int(sum(matrix.todense()[:,vocab[w]] > 0)))+'/'+str(n))]
  return result
    
def load_model(doc_size,ngram=1,models_folder=MODELS_FOLDER):
  if ngram:
    extension='%d_%d'%(ngram,doc_size)
  else:
    extension='%d'%(doc_size)
    
  vectorizer=pickle.load(open(os.path.join(models_folder,'vectorizer_%s' % extension),'rb'))
  matrix=pickle.load(open(os.path.join(models_folder,'matrix_%s' % extension),'rb'))
  bookids=pickle.load(open(os.path.join(models_folder,'bookids_%s' % extension),'rb'))
  vocab=pickle.load(open(os.path.join(models_folder,'vocab_%s' % extension),'rb'))
  return vectorizer,matrix,bookids,vocab
  #vectorizer,matrix,bookids,vocab=load_model(10)
    
def most_common_words(book_ids,transformed_matrix,id2word,n=20,wanted_ids=None):
  for b,v in zip(book_ids,transformed_matrix.todense()):
    if wanted_ids != None:
      if b not in wanted_ids:
        continue
    word_ids=np.argsort(v).tolist()[0][::-1][:n]
    words=[id2word[x] for x in word_ids]
    scores=[v[0,x] for x in word_ids]
    print('\n%d:'%b)
    for w,s in zip(words,scores):
      print("%.3f - %s"%(s,w))
      
def main(argv):
  
  if not os.path.exists(MODELS_FOLDER):
    os.mkdir(MODELS_FOLDER)
    
  if not os.path.exists(TOKENIZED_PATH):
    os.mkdir(TOKENIZED_PATH)
  
  tokenize=tokenize1
  ngram=1
  
  if len(argv) > 2:
    bookids=range(int(argv[1]),int(argv[2]))
    df_meta = pickle.load(open('../res/metadata_extended.pickle', 'rb'))
    bookids=[b for b in bookids if b in df_meta.index]
    if len(argv) > 3:
      if argv[3]=='3':
        tokenize=tokenize3
        ngram=3
      elif argv[3]=='2':
        tokenize=tokenize2
        ngram=2
  else:
    bookids=[1,11,12,30,74,76,82,161,1342,1661]
    bookids=list(range(40,50))+[161,1342,1661]
  
  start=time.time()
  docs=get_docs(bookids)

  vectorizer=CountVectorizer(#ngram_range = (1, 2),
                             min_df=3,
                             max_features=50000,
                             tokenizer=tokenize,
                             preprocessor=preprocess)
  matrix=vectorizer.fit_transform(docs)
  
  end=time.time()
  id2word=dict()
  for k,v in vectorizer.vocabulary_.items():
    id2word[v]=k

  
  print('Total time: %.3f'%(end-start))
  print('Total books: %d'%len(BOOKIDS))
  print('%.3f per book.'%((end-start)/len(BOOKIDS)))
  
  most_common_words(BOOKIDS,matrix,id2word,5,wanted_ids=[1,10,11,12,30,74,76,82,161,1342,1661])
  
  extension='%d_%d'%(ngram,len(BOOKIDS))
  
  print('Saving matrix',end='')
  start=time.time()
  pickle.dump(matrix,open(os.path.join(MODELS_FOLDER,'matrix_%s' % extension),'wb'))
  print('\rMatrix saved: %.3f'%(time.time()-start))
  
  print('Saving bookids',end='')
  start=time.time()
  pickle.dump(BOOKIDS,open(os.path.join(MODELS_FOLDER,'bookids_%s' % extension),'wb'))
  print('\rBookids saved: %.3f'%(time.time()-start))
  
  print('Saving vocab',end='')
  start=time.time()
  pickle.dump(vectorizer.vocabulary_,open(os.path.join(MODELS_FOLDER,'vocab_%s' % extension),'wb'))
  print('\rVocab saved: %.3f'%(time.time()-start))
  
  print('Saving vectorizer',end='')
  start=time.time()
  pickle.dump(vectorizer,open(os.path.join(MODELS_FOLDER,'vectorizer_%s' % extension),'wb'))
  print('\rVectorizer saved: %.3f'%(time.time()-start))
  
  print('Saving matrix small',end='')
  start=time.time()
  pickle.dump(csr_matrix(matrix,dtype=np.int32),open(os.path.join(MODELS_FOLDER,'matrix_small_%s' % extension),'wb'))
  print('\rMatrix small saved: %.3f'%(time.time()-start))

if __name__=='__main__':
  main(sys.argv)

