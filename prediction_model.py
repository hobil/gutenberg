"""
Creates Model object which contains the classifier (NB, tree, or random forest) from sklearn. It has some extra functionalities implemented, such as predicting multiple labels with their probabilities instead predicting one label,.
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, roc_auc_score, recall_score,f1_score
from sklearn.preprocessing import scale
import pickle


class Model(object):

  def create_model(self, model_type, **kwargs):
    
    def create_tree_params(**kwargs):
      params = dict()
      if 'max_depth' in kwargs:
        params['max_depth'] = kwargs['max_depth']
      else:
        params['max_depth'] = None
  
      if 'max_features' in kwargs:
        params['max_features'] = kwargs['max_features']
      else:
        params['max_features'] = 'auto'
        
      if 'class_weight' in kwargs:
        params['class_weight'] = kwargs['class_weight']
      else:
        params['class_weight'] = None
      
      return params
  
    if model_type == 'm':
      return MultinomialNB()
    
    elif model_type == 'g':
      return GaussianNB()
    
    elif model_type == 'd':
      return DecisionTreeClassifier(**create_tree_params(**kwargs))
    
    elif model_type == 'r':
      params = create_tree_params(**kwargs)
      params['n_jobs'] = -1
      if 'bootstrap' in kwargs:
        params['bootstrap'] = kwargs['bootstrap']
      else:
        params['bootstrap'] = True
      if 'n_estimators' in kwargs:
        params['n_estimators'] = kwargs['n_estimators']
      else:
        params['n_estimators'] = 1
      return RandomForestClassifier(**params)
    
    elif model_type == 'a':
      params = dict()
      params['base_estimator'] = self.create_model(kwargs['base_estimator'], **kwargs)
      if 'n_estimators' in kwargs:
        params['n_estimators'] = kwargs['n_estimators']
      else:
        params['max_depth'] = None
      return AdaBoostClassifier(**params)
    
  def __init__(self, model_type, attribute, feature_names = None, **kwargs):
    self.model = self.create_model(model_type, **kwargs)
    self.model_type = model_type
    self.attribute = attribute
    self.feature_names = feature_names
      
  def fit(self, X, y):
    self.model.fit(X, y)
    self.compute_class_feature_importance(X,y,self.model.feature_importances_)
  
  def predict_one(self, X,y=None,statistics=False,labels=None):
    pred = self.model.predict(X)
    if statistics:
      self.compute_statistics(y,pred,labels)
    return pred
  
  def predict_proba(self,X):
    return self.model.predict_proba(X)

  def predict(self,X, n=None, min_prob = 0.0):
    
    predictions = self.model.predict_proba(X)
    if set([True,False]) == set(self.model.classes_):
      # binary model
      # True label is in the second column, because the labels are sorted
      idx_True = 1
      return predictions[:,idx_True]
      return predictions[idx_True]
    else:
      # multiclass model
      predictions_filtered = [sorted([(name,score) for name,score in zip(self.model.classes_,prediction) if score > min_prob],key = lambda x: x[1], reverse = True) for prediction in predictions]
      return predictions_filtered[:n]
  
  def get_parameter_importance(self,n_parm = None, min_importance = -np.inf):
    if self.feature_names is None:
      print("No feature names, cannot compute feature importances.")
      return
    featname_importance = zip(self.feature_names,self.model.feature_importances_)
    featname_importance = [(featname,importance) for featname,importance in featname_importance if importance > min_importance]
    featname_importance = sorted(featname_importance, key = lambda x: x[1], reverse = True)
    return featname_importance[:n_parm]
    
  def compute_statistics(self,y_true,y_pred,labels=None):
    s = dict()
    if labels is None:
       labels = sorted(list(set(y_true)))
      
    cm = confusion_matrix(y_true,y_pred,labels)
    print("Confusion matrix:")
    cm=pd.DataFrame(cm,columns=labels,index=labels)
    s['cm']=cm
    print(cm)
    
    print("Accuracy:")
    acc = accuracy_score(y_true,y_pred)
    s['acc']=acc
    print(acc)
    
    print("F macro")
    f_macro = f1_score(y_true,y_pred,average='macro')
    s['f_macro']=f_macro
    print(f_macro)
    
    print("F micro")
    f_micro = f1_score(y_true,y_pred,average='micro')
    s['f_micro']=f_micro
    print(f_micro)

    if len (labels) == 2:
      print("F binary")
      print(f1_score(y_true,y_pred,average='binary')) 
    return s
  
  def compute_class_feature_importance(self,X, Y, feature_importances):
    N, M = X.shape
    X = scale(X)

    out = {}
    for c in set(Y):
        out[c] = dict(
            zip(range(N), np.mean(X[Y==c, :], axis=0)*feature_importances)
        )
    outdf=pd.DataFrame(out).T
    outdf.columns=self.feature_names
    self.class_feature_importance = outdf
  
  def save(self, filename = None):
    if filename == None:
      filename = '../res/models/%s_model.pickle' % self.attribute
    pickle.dump(self, open(filename, 'wb'))
  
  @classmethod
  def load(self, filename):
    return pickle.load(open(filename, 'rb'))
  
def show_confusion_matrix(self, y_true,y_pred,labels=None):
  if labels is None:
    labels = sorted(list(set(y_true)))
  self.confusion_matrix_abs = pd.DataFrame(confusion_matrix(y_true,y_pred),index = labels, columns = labels)
  total = self.confusion_matrix_abs.sum()
  self.confusion_matrix = self.confusion_matrix_abs / total
  

    
    

  
  
  

class Evaluator(object):
  def __init__(self,y_true,y_pred):
    self.y_true = y_true
    self.y_pred = y_pred
    self.roc_auc_score = roc_auc_score(y_true,y_pred)
    self.accuracy_score = accuracy_score(y_true,y_pred)
    self.matthews_corrcoef = matthews_corrcoef(y_true,y_pred)
    self.confusion_matrix_abs = confusion_matrix(y_true,y_pred)
    total = self.confusion_matrix_abs.sum()
    self.confusion_matrix = self.confusion_matrix_abs / total
    'TN','FN','TP','FP'
    self.TN = self.confusion_matrix[0,0] 
    self.FN = self.confusion_matrix[1,0] 
    self.TP = self.confusion_matrix[1,1] 
    self.FP = self.confusion_matrix[0,1]
    self.recall_score = recall_score(y_true,y_pred)
    self.specificity = self.TN / (self.TN + self.FP)
    self.balanced_accuracy = (self.recall_score + self.specificity)/2
  
  def __str__(self):
    return str(self.accuracy_score)
