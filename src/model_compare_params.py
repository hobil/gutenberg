import time
from models import load_fvs, split_into_train_and_test
import pickle
import numpy as np
from prediction_model import Model, Evaluator
from metadata import load_metadata
from sklearn.metrics import confusion_matrix
import pandas as pd
import sys

subjects = ['adventure_stories', 'love_stories', 'short_stories', 'historical_fiction', 'science_fiction', 'poetry', 'drama', 'detective_and_mystery_stories']

def save_models(a,model_type = 'r', fv_type = 'tfidf', depth = 15, estimators = 200, size = '2k', sampling = None, combined = False, max_features = 'auto'):
  #df_meta = load_metadata()
  df = load_fvs(fv_type, size = size, combined=combined)
  print("df shape: (%d,%d)"%df.shape)
  #idx = df_meta.query('PQRSTZ').index
  #df = df.ix[idx]
  X_train, y_train, X_test, y_test = split_into_train_and_test(df, a, min_occurence = 5,sampling = sampling)
  m = Model(model_type = model_type, attribute = a, n_estimators = estimators, max_depth = depth, feature_names = df.columns, class_weight = 'auto')
  m.fit(X_train, y_train)
  y_pred = m.predict(X_test)
  print("\n\n%s,%s,%s,%s"%(model_type,fv_type,size,str(combined)))
  statistics = m.compute_statistics(y_test,y_pred)
  #pickle.dump(m,open('../res/models/%s_model.pickle'%a,'wb'))
  return statistics
  
  

def main(**args):
  
  model_types  = ['m','g','d','r','r','r','m','g','d','r','r','r','m','g','d','r','r','r']
  sizes = ['500','500','500','500','500','500','2k','2k','2k','2k','8k','8k','8k','8k']
  fv_types = ['bin','tfidf','tfidf','text_feats','tfidf','tfidf','bin','tfidf','tfidf','text_feats','tfidf','tfidf','bin','tfidf','tfidf','text_feats','tfidf','tfidf']
  combineds = [False,True,True,False,False,True,False,True,True,False,False,True,False,True,True,False,False,True]
  
  
  if 'attribute' in args:
    a = args['attribute']
  else:
    a = 'country'
  if 'model_type' in args:
    model_type = args['model_type']
  else:
    model_type = 'r'
  if 'fv_type' in args:
    fv_type = args['fv_type']
  else:
    fv_type = 'tfidf'
  if 'depth' in args:
    depth = int(args['depth'])
  else:
    depth = 20
  if 'max_features' in args:
    max_features = args['max_features']
  else:
    max_features = 'auto'
  if 'sampling' in args:
    sampling = args['sampling']
  else:
    sampling = None
  if 'n_estimators' in args:
    n_estimators = int(args['n_estimators'])
  else:
    n_estimators = 200
  if 'combined' in args:
    combined = bool(args['combined'])
  else:
    combined = False
  
  results = dict()
  for model_type,size,fv_type,combined in zip(model_types,sizes,fv_types,combineds):
    results[(model_type,size,fv_type,combined)] = save_models(a, fv_type = fv_type ,model_type = model_type, depth = depth, sampling = sampling, combined = combined, max_features = max_features, estimators = n_estimators,size=size )
  pickle.dump(results,open('../res/models/results_%s_%d.pickle'%(a,depth),'wb'))
  
if __name__ == '__main__':
  parm = dict()
  for x in sys.argv[1:]:
    parm[x.split('=')[0].strip()] = x.split('=')[1].strip()
  main(**parm)


