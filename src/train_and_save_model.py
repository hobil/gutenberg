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
  y_pred = m.predict_one(X_test,y_test,True)

  labels = set(y_train)
  print("---%d---"%depth)
  print("Total accuracy: %.3f\n"%np.average(y_pred == y_test))
  accs = []
  for l in labels:
    tp = np.sum(y_pred[y_test == l] == y_test[y_test == l])
    p = np.sum(y_test == l)
    accs.append(tp/p)
    print(l,"%d / %d, %.3f"%(tp,p,tp/p))
  print("\navg accuracy: %.3f\n"%np.average(accs))
  #cf = confusion_matrix(y_test,y_pred,labels = list(labels)) / len(y_test)
  #print(pd.DataFrame(cf,columns = labels, index = labels).round(3))
  """
  ev = Evaluator(y_test,y_pred)
  print()
  print(a,".%3f"%ev.balanced_accuracy,"%.3f"%ev.recall_score)
  print(ev.confusion_matrix.round(3))
  print("=========================")
  """
  pickle.dump(m,open('../res/models/%s_model.pickle'%a,'wb'))
  
  

def main(**args):
  
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
    depth = 10
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
    n_estimators = 100
  if 'combined' in args:
    combined = bool(args['combined'])
  else:
    combined = False

  save_models(a, fv_type = fv_type ,model_type = model_type, depth = depth, sampling = sampling, combined = combined, max_features = max_features, estimators = n_estimators )

if __name__ == '__main__':
  parm = dict()
  for x in sys.argv[1:]:
    parm[x.split('=')[0].strip()] = x.split('=')[1].strip()
  main(**parm)

