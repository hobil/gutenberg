"""
Loads metadata if available. Otherwise downloads the PG catalog."""

import os
import pickle
import numpy as np
import pandas as pd
from .metainfo import readmetadata

PICKLED_FILE='../res/metadata_extended.pickle'

subjects1 = ['adventure stories', 'love stories', 'short stories', 'historical fiction', 'science fiction', 'poetry', 'drama', 'detective and mystery stories']

#subjects_old = ['history', 'juvenile_fiction', 'united_states', 'periodicals', 'social_life_and_customs', '19th_century', 'great_britain', 'england', 'biography', 'short_stories', 'description_and_travel', 'science_fiction', 'translations_into_english', 'conduct_of_life', 'world_war, 1914-1918', 'poetry', 'adventure_stories', 'france', 'english_wit_and_humor', 'drama', 'love_stories', 'politics_and_government', 'detective_and_mystery_stories', 'juvenile_literature', 'historical_fiction', 'young_women', 'civil_war, 1861-1865', 'children', 'western_stories']

epochs = ['Ancient','Medieval','Renaissance','Baroque','Enlightenment','Romanticism','Realism','Late 19th','Modern']
epochs_short = ['Ancient','Medieval','Renaiss','Baroque','Enlight','Romant','Realism','Late 19','Modern']
# maps LCC parent classes to 6 super categories
mapping = {
    'A' : 'General Works',
    'B' : 'Philosophy Psychology Religion',
    'C' : 'History and Geography',
    'D' : 'History and Geography',
    'E' : 'History and Geography',
    'F' : 'History and Geography',
    'G' : 'History and Geography',
    'H' : 'Social Science and Arts',
    'J' : 'Social Science and Arts',
    'K' : 'Social Science and Arts',
    'L' : 'Social Science and Arts',
    'M' : 'Social Science and Arts',
    'N' : 'Science and Technology',
    'P' : 'Language and Literature',
    'Q' : 'Science and Technology',
    'R' : 'Science and Technology',
    'S' : 'Science and Technology',
    'T' : 'Science and Technology',
    'U' : 'Science and Technology',
    'V' : 'Science and Technology',
    'Z' : 'General Works'
    }

super_categories = ['General Works','Philosophy Psychology Religion','History and Geography','Social Science and Arts','Science and Technology','Language and Literature']
super_categories_short = ['GenWorks','PhiPsyRel','HistGeo','SocSci&Art','Sci&Tech','Lang&Lit']

pos_mapping = {
    'ADJ' : 'adjective',
    'ADP' : 'adposition',
    'ADV' : 'adverb',
    'CONJ': 'conjunction',
    'DET' : 'determiner',
    'NOUN': 'noun',
    'NUM' : 'numeral',
    'PART': 'particle',
    'PRON': 'pronoun',
    'PROPN':'proper noun',
    'PUNCT':'punctuation',
    'VERB':'verb',
'MD':'MD -- modal verb',
'VB':'VB -- verb in base form',
'VBD':'VBD -- past tense',
'VBG':'VBG -- gerund or pres. part.',
'VBN':'VBN -- past participle',
'VBP':'VBP -- pres. tense, non-3rd p. singl.',
'VBZ':'VBZ -- pres. tense, 3rd p. singl.',
    
}

def load_metadata():
  """
  Returns the content of the pickled metadata file if available. Otherwise, compute the metadat.
  """
  if not os.path.exists(PICKLED_FILE):
    save_metadata()
  return pickle.load(open(PICKLED_FILE,'rb'))

def save_metadata():
  """
  Computes the metadata and saves it for future reference.
  """
  
  def extract_lcc_class_simple(LCC_set):
    result=set([mapping[x[0]] for x in LCC_set])
    if result is not None:
      if len(result) == 1:
        return tuple(result)[0]
    return None
  
  def extract_lcc_class_simple2(LCC_set, super_category):
    result=set([mapping[x[0]] for x in LCC_set])
    return super_category in result
    
  def extract_subject_tags(subject_set,tags_to_be_kept):
    if subject_set is not None:
      if len(subject_set)==1:
        subj= list(subject_set)[0]
        if subj in tags_to_be_kept:
          return subj
  
  def epoch_names(year):
    """
    Extracts epoch.
    """
    if year < 500:
      return 'Ancient'
    elif year <1450:
      return 'Medieval'
    elif year <1550:
      return 'Renaissance'
    elif year <1685:
      return 'Baroque'
    elif year <1750:
      return 'Enlightenment'
    elif year <1800:
      return 'Romanticism'
    elif year <1850:
      return 'Realism'
    elif year <1900:
      return 'Late 19th'
    elif year >=1900:
      return 'Modern'
  
  def epoch(year):
    """
    Extracts epoch and assigns a number to it instead of name.
    """
    if np.isnan(year):
      return None
    elif year < 500:
      return 0
    elif year <1450:
      return 1
    elif year <1550:
      return 2
    elif year <1685:
      return 3
    elif year <1750:
      return 4
    elif year <1800:
      return 5
    elif year <1850:
      return 6
    elif year <1900:
      return 7
    elif year >=1900:
      return 8
  
  def lcsh_simple(LCSH_set):
    """
    Splits each LCSH subject on dash and comma
    """
    result=set()
    for s in LCSH_set:
      arr=s.replace('--',',').split(',')
      for a in arr:
        result.add(a.strip().lower())
    return result
  
  def is_p(LCC_set):
    """
    Returns if the book has a 'P' class.
    """
    for lcc in LCC_set:
      if lcc[0] == 'P':
        return True
    return False
  
  def is_prsz(LCC_set):
    """
    Returns is the book is classical fiction (belles lettrez).
    Includes only American, English and children literature.
    """
    for lcc in LCC_set:
      if lcc in ['PR','PS','PZ']:
        return True
    return False
  
  def is_pqrstz(LCC_set):
    """
    Returns is the book is classical fiction (belles lettrez).
    """
    for lcc in LCC_set:
      if lcc in ['PR','PS','PZ', 'PT','PQ']:
        return True
    return False
  
  def has_subject(lcsh_simple_set, subject):
    """
    Returns if a given subject is in the set of subjects.
    """
    return subject in lcsh_simple_set
  
  def american_british(LCC_set):
    """
    Extracts PS and PR tags from LCC set.
    """
    if 'PS' in LCC_set:
      if 'PR' not in LCC_set:
        return 'American'
    elif 'PR' in LCC_set:
      if 'PS' not in LCC_set:
        return 'British'
    return None
  
  ####
  # This section excludes some documents not interesting for our analysis
  ####  
  BLACKLIST_IDS=[303,10742,28898,29441,29156,22427,29424,23398,36755,33627,35921,3206,45265]
  BLACKLIST_IDS+=[682, 812, 2585, 744, 2583, 633, 2586, 634, 2584, 69, 63, 302,212]#mathematical constants
  BLACKLIST_IDS+=[35237,44740,45289, 30243, 29459, 35471, 29430, 29467, 29431, 23594, 9454, 45023, 28760, 8700, 44743, 49589,22335,38115,32967,32255,33108,41430,46392, 23446, 20086, 15809, 19177] #ILLUSTRATIONS
  BLACKLIST_IDS+=[6300, 13536, 13571, 23524, 25525, 25533, 25541, 25584, 25717, 25808, 25902, 25903, 26770, 28076, 28701, 28747, 28803, 28821, 28822, 28823, 28850, 28858, 28859, 28871, 28876, 28879,28884,28896, 28905, 28959, 28969, 28970, 28972, 29004, 29250, 29381, 29434, 32937, 37228, 43036] #Indexes
  BLACKLIST_IDS+=[997,998,999,1000,43679,47051,49110,48768,5424,26413,41810, 23984, 35815,27737,48232,50503,29214,36548] # not english
  BLACKLIST_IDS+=[41697,41952,25738,27468,31830,39357,40311,4656,4542,37504,21783,26513,21054,88,115,29,37,11800,54084,52841,51671,51155,251,36073,37237,34924,42494,249,9172,36673,43648,43776,44694,48043,52371,17719,17776,31543,33013,40900,41217,44527,52230,28860,38390,32309,34815,43857,21595,32050,34554,32309,32426,10625,19606,29211,36021,43305,43737,46692,53911,23147,31565,45077, 255, 3204, 36877] #komisch zeug
  
  BLACKLIST_IDS+=[673,29765,13188,22] # dictionaries
  BLACKLIST_IDS+=[279]
  BLACKLIST_IDS+=list(range(11800,11857)) # catalogs of copyright renewals
  BLACKLIST_IDS+=[6161,6348,6349,6419,6420] # other copyrights
  BLACKLIST_IDS+=[44621,51836, 23053, 19394] # motion pictures, catalog of copyright
  BLACKLIST_IDS+=[117,156,276,17421,17422,17423] # music
  
  BLACKLIST_IDS+=[13651,13688,14359,15751,16844,17247,17606,19553,20442,22690,24402,27477,27606,31016,31183,31639,33034,33102,33183,35272,35273,35274,35842,36411,36434,36569,36598,36649,36960,37055,37460,37558,37658,37796,38301,38658,39089,39329,39464,39677,39818,40118,40536,40730,41290,41321,42121,42420,42468,42652,42977,43384,43385,43386,44014,44090,44556,44988,44995,45433,45516,45737,45744,46506,46507,46619,46620,47779,49620,49919,51672] # other catalogs
  BLACKLIST_IDS+=[247, 248, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 37683, 38538, 38699, 38700, 44709,31731,41975] # dictionaries
  BLACKLIST_IDS+=[48611] # commercial catalog
  BLACKLIST_IDS+=[44661, 44662, 44663, 45813, 46349, 50428] # english wit and humor, pictorial
  BLACKLIST_IDS+=[1359,22228,22636,25782,26794,29152,29560,30580,30976,33477,33991,34316,39819,40944,47436,50336] # high POS rate
  BLACKLIST_IDS += [28409, 33374, 12701, 11262, 14557, 24146, 26752, 17424] # no text inside
  BLACKLIST_IDS += list(range(7901, 7925))
  BLACKLIST_IDS += [51104] # wrong download
  BLACKLIST_IDS += [32458] # short poem
  ####
  

  # loads the data from catalog
  md_dict = readmetadata()
  # creates data frame
  md_df = pd.DataFrame.from_dict(md_dict,orient='index').set_index('id')
  # deletes empty repositories
  md_df = md_df[md_df.formats != {}]
  #filter out non text files
  md_text_df = md_df[md_df['type']=='Text']
  # take books which are in English only
  df = md_text_df[md_text_df['language'].apply(lambda x:x==['en'])]
  # do not inluce books from BLACKLIST
  blacklist = [b for b in BLACKLIST_IDS if b in df.index]
  df = df.drop(blacklist)

  # add epoch indexed by integers
  df['epoch'] = df['authoryearofbirth'].apply(epoch)
  # add epoch with names
  df['epoch_names'] = df['authoryearofbirth'].apply(epoch_names)
    
  # add flags regarding lcc to the metadata dataframe
  df['lcc_class_simple']=df['LCC'].apply(extract_lcc_class_simple)
  df['country'] = df.LCC.apply(american_british)
  df['lcsh_simple']=df.subjects.apply(lcsh_simple)
  df['P']=df.LCC.apply(is_p)
  df['PRSZ']=df.LCC.apply(is_prsz)
  df['PQRSTZ']=df.LCC.apply(is_pqrstz)
  
  # adds a boolean flag for 
  for s in subjects1:
    print("doing %s"%s)
    s_with_underscore = s.replace(' ','_')
    df[s_with_underscore] = df.lcsh_simple.apply(lambda x: has_subject(x,s))
  
  # adds a boolean flag for all 6 super categories
  for cat in super_categories:
    df[cat.lower().replace(' ','_')] = df.LCC.apply(lambda x: extract_lcc_class_simple2(x,cat))
  
  # save metadata to a pickle file
  pickle.dump(df,open(PICKLED_FILE,'wb'))
  return df

if __name__ == '__main__':
  save_metadata()

