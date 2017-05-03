import sys
import os
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import spacy
import networkx as nx

print("loading parser")
nlp = spacy.en.English()

def is_person(entity, d):
  """
  For given entity returns if it is an entity of type person.
  """
  return d[entity.start].ent_type_ == 'PERSON'


def clean_entity(e, add_titles = True):
  """
  Adds title and cleans the entity from unwanted symbols in the beginning and at the end.
  """

  if add_titles and e.start != 0:
    # not the first word in the book
    l = len(e)
    previous_word = e[-(l+1)]
    
    if previous_word.text in ('Mr.', 'Mrs.', 'Miss', 'Mister'):
      e = previous_word.text.strip() +" " + e.text.strip()
      #print("added",e)
    else:
      e=e.text.strip()
  else:
    e=e.text.strip()
  
  # strip possesive 's at the end of the word
  e = e.split("'")[0]
  # strip dash
  e = e.split("--")[0]
  # strip interpunction if part of entity
  chars = '\'"?!--;()[]|-{}., \t\n'
  e = e.strip(chars)
  return e

def filter_names(names_sentid, max_characters = 20):
  c = Counter([x[0] for x in names_sentid]).most_common(max_characters)
  big_characters = set([name for name,val in c])
  filtered_names_sentid = [(n,sent_id) for n,sent_id in names_sentid if n in big_characters]
  return filtered_names_sentid

def check_dict(dictionary, entry, default):
  """
  Dummy functions which enters a default value to a dictionary if it is not there. For more convenient working.
  """
  if entry not in dictionary:
    dictionary[entry] = default   

def character_score(names_sentid, max_sent_dist=0, f = lambda x: 0.5**x):
  """
  Computes interactions (dict) for every character
  """
  interactions = dict()
  interactions_sum = dict()
  for k,(n1,sent1) in enumerate(names_sentid):
    for n2,sent2 in names_sentid[k+1:]:
      if sent2 - sent1 > max_sent_dist:
        break
        # create entry if a new name
      check_dict(interactions, n1, dict())
      check_dict(interactions, n2, dict())
      check_dict(interactions_sum, n1, 0.0)
      check_dict(interactions_sum, n2, 0.0)
      check_dict(interactions[n1], n2, 0)
      check_dict(interactions[n2], n1, 0)
      
      score = f(np.abs(sent2-sent1))
      interactions[n1][n2] += score
      interactions[n2][n1] += score
      interactions_sum[n1] += score
      interactions_sum[n2] += score
  return interactions, interactions_sum

def create_token_sent_dict(d):
  sent_dict = dict()
  for n,s in enumerate(d.sents):
    for x in range(s.start,s.end):
        sent_dict[x] = n
  return sent_dict

def group_names_together(names_sentid,min_occurence_of_super_name = 6,allow_aliasing = True):
  """
  Group the names of the same character together. The name belongs to the same character if it is a subset of it.
  """
  
  def split_name(name):
    arr = set(name.split())
    arr2 = []
    for a in arr:
      arr2 += (a.split("-"))
    return [a for a in arr2 if a != '']
  
  def is_name_contained(name,other_name):
    if name == other_name:
      return False
    other_name_split = split_name(other_name)
    return all([n in other_name_split for n in split_name(name)])

  if not allow_aliasing:
    return names_sentid
  name_set = set([n for n,sentid in names_sentid])
  c = Counter([n for n,sentid in names_sentid])
  #super_names = dict()
  alias = dict()
  for n in name_set:
    alias[n] = n
  names_by_length = sorted(name_set, key = len, reverse = True)
  
  for name in names_by_length:
    contained_in = [other_name for other_name in name_set if is_name_contained(name,other_name) and c[other_name] >= min_occurence_of_super_name]
    if len(contained_in) == 1:
      super_name = contained_in[0]
      alias[name] = super_name
      # TODO: remove print
      print(name," -> ",super_name)
      name_set.remove(name)
    elif len(contained_in) > 1:
      # is it can't be decided because of titles, choose the one without title
      contained_in = set(contained_in)
      without_title = set()
      for x in contained_in:
        if x.startswith('Miss') or x.startswith('Mr.') or x.startswith('Mrs.') or x.startswith('Miss'):
          continue
        else:
          without_title.add(x)
      
      if len(without_title) == 1:
        super_name = list(without_title)[0]
        alias[name] = super_name
        # TODO: remove print
        print("SAVED:",name," -> ",super_name)
        name_set.remove(name)
      #else:
        #print("MISMATCH:",name,contained_in)
  return [(alias[n],sent_id) for n,sent_id in names_sentid]

def analyze_book(d, n_characters = 20,max_sent_dist = 1,min_occurence_of_super_name=6,allow_aliasing=True):
  
  # assign sentence_id to each token
  sent_dict = create_token_sent_dict(d)
  
  # saves tuple (cleaned_entity, sentence_id)
  # cleaning fixes imperfections of the spacy tagger
  names_sentid = [(clean_entity(entity), sent_dict[entity.start]) for entity in d.ents if is_person(entity, d)]
  # filter on only person names and assign sentence_id
  
  names_sentid_grouped = group_names_together(names_sentid,min_occurence_of_super_name=min_occurence_of_super_name,allow_aliasing=allow_aliasing)
  
  character_counts = Counter([n for n,sent_id in names_sentid_grouped])
  
  names_sentid_filtered = filter_names(names_sentid_grouped, max_characters=n_characters)
  
  interactions, interactions_sum = character_score(names_sentid_filtered, max_sent_dist = max_sent_dist, f = lambda x: 0.5**x)
  
  return interactions,character_counts


def create_graph(interactions):
  G=nx.Graph()
  for i in interactions:
    G.add_node(i)
    
  for n1,vals in interactions.items():
    for n2,v in vals.items():
      G.add_edge(n1,n2,weight=v)
  return G

def plot_graph(interactions,character_counts, layout = nx.spring_layout, min_node_size = 200, max_edge_size = 20, font_size = 18):
  G = create_graph(interactions)
  fig,ax = plt.subplots(figsize=(30,30))
  # nodes
  node_weights = np.array([character_counts[n] for n in G.nodes()])
  arg_sorted = np.argsort(node_weights)[::-1]
  
  #ensure all nodes have at least min_node_size
  min_node_weight = node_weights.min()
  if min_node_weight < min_node_size:
    node_sizes = node_weights * min_node_size / min_node_weight
  else:
    node_sizes = node_weights
    
  if layout == nx.shell_layout:
    n_middle = 1

    shells = [[G.nodes()[i] for i in arg_sorted[:n_middle]],[G.nodes()[i] for i in range(len(G.nodes())) if i !=arg_sorted[0]]]
    pos = layout(G)
    
    pos=layout(G,shells)
  elif layout == nx.spring_layout:      
    pos = layout(G,iterations = 500, weight = None)
  else:
    pos = layout(G)
  
  #sorted_names = sorted(interactions_sum,key = lambda x:interactions_sum[x])[::-1]
  nx.draw_networkx_nodes(G,pos,with_labels = True, node_color = '#b6e3ff', node_size = node_sizes, linewidths = 0,ax=ax)
  nx.draw_networkx_labels(G,pos,font_size=font_size,font_family='sans-serif',ax=ax)
  
  # edges
  max_edge_weight = max([x[2]['weight'] for x in G.edges(data = True)])
  if max_edge_weight > max_edge_size:
    edge_scale = max_edge_size / max_edge_weight
  else:
    edge_scale = 1.0
  
  for e in G.edges(data = True):
    weight = e[2]['weight']
    edge_size = weight * edge_scale
    nx.draw_networkx_edges(G,pos,edgelist=[e], width=edge_size, edge_color = "#b3b3b3",ax=ax,)
  return fig,ax

"""
sherlock = nlp(load_and_filter_document(1661))
us = nlp(load_and_filter_document(1))
d = harry
d = alice
d = sherlock
name = 'Alice'
name = 'Harry'
"""

def print_usage():
  """
  Prints usage if user entered parameters in a wrong form.
  """
  
  print("Wrong parameter, usage:")
  print("python3 character_interaction.py -f [filename]")
  print("OR")
  print("python3 character_interaction.py -f [filename] [output_file]")
  return

def main(args):
  # parse the input and ouput parameters
  if len(args) < 2:
    print_usage()
    return
  elif len(args) == 2:
    if args[0] != '-f':
      print_usage()
      return
    elif not os.path.exists(args[1]):
      print("File %s does not exist." % args[1])
      return
    else:
      doc = open(args[1]).read()
      OUTPUT_FILE_NAME = 'output.pdf'
  elif len(args) == 3:
    if args[0] != '-f':
      print_usage()
      return
    elif not os.path.exists(args[1]):
      print("File %s does not exist." % args[1])
      return
    else:
      doc = open(args[1]).read()
      OUTPUT_FILE_NAME = args[2]
  else:
    print_usage()
  
  # extract entities from text
  d = nlp(doc)
  # select person entities and analyze interactions
  interactions, character_counts  = analyze_book(d)
  # save a grpah with interactions to pdf
  fig, ax = plot_graph(interactions,character_counts)
  fig.savefig(OUTPUT_FILE_NAME)
  return fig,ax, interactions,character_counts

if __name__ == '__main__':
  main(sys.argv[1:])
