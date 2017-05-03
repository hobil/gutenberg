"""
This module loads the English text books based on metadata info. It stores the information about already downloaded books not to download them multiple times.

UPDATE 7.4.2017: PG has changed the way to download the book data, these files seem to be not available at the moment with error 403.
To download book texts, use of the ways mentioned in http://www.gutenberg.org/wiki/Gutenberg:Information_About_Robot_Access_to_our_Pages.
""" 

from __future__ import print_function

import os
import pandas as pd
import requests
import logging
import zipfile
import metainfo as metainfo

ZIPDIR="../res/zips/"
TXTDIR="../res/txts/"
URL_BEGIN="http://www.gutenberg.lib.md.us"
LOGGER_DIR="../res/log/"

def get_logger():
  """Returns a logger. The logger is saved in the current directory as "logger.log". """
  logging.basicConfig(filename=LOGGER_DIR+"download_books.log",
                      filemode='w',
                      format='%(levelname)s %(message)s',
                      level=logging.INFO)
  return logging.getLogger()

def get_url(book_id):
  if book_id < 10:
    result="/0/"+str(book_id)+"/"+str(book_id)
  else:
    i=str(book_id)
    result=""
    while len(i) > 1:
      result+="/"+i[0]
      i=i[1:]
    result+="/"+str(book_id)+"/"+str(book_id)
  return URL_BEGIN+result

if not os.path.exists(ZIPDIR):
    os.makedirs(ZIPDIR)
if not os.path.exists(TXTDIR):
    os.makedirs(TXTDIR)
if not os.path.exists(LOGGER_DIR):
    os.makedirs(LOGGER_DIR)
    
logger=get_logger()

md_dict=metainfo.readmetadata()
#54034 repositories in 22.1.2017
md_df=pd.DataFrame.from_dict(md_dict,orient='index').set_index('id')
md_df = md_df[md_df.formats != {}]
#filter out empty repositories
#53962

#filter out non text files
md_text_df=md_df[md_df['type']=='Text']
#52718

# take books written only in pure english
df=md_text_df[md_text_df['language'].apply(lambda x:x==['en'])]
ids=list(df.index)
n=len(ids)
not_found=[]
downloaded=0
already_available=0
already_available_set=os.listdir(ZIPDIR)
already_available_set=frozenset([int(b.split(".")[0].split("-")[0]) for b in already_available_set])

for i,book_id in enumerate(ids):
  """ if the zip file already exists, skip """
  if book_id in already_available_set:
    already_available+=1
    logger.info("Book_id %d already downloaded."%book_id)
    print("\r%0.2f%%"%(100.*i/n),end='')
    continue
  
  url=get_url(book_id)
  suffix=['.zip','-0.zip','-8.zip']
  for j,suf in enumerate(suffix):
    zip_file=ZIPDIR+str(book_id)+suf
    """ download the zip file"""
    r = requests.get(url+suf)
    """ if download not successful try another format """
    if r.status_code==200:
      """save the zip file"""
      with open(zip_file, 'wb') as f:
        f.write(r.content)
      """extract the zip file"""
      zfile=zipfile.ZipFile(zip_file)
      zfile.extractall(TXTDIR)
      downloaded+=1
      """show the progress in %"""
      print("\r%0.2f%%"%(100.*i/n),end='')
      logger.info("Book_id %d successfully extracted."%book_id)
      break
    elif r.status_code==404:
      logger.info("URL %s not found."%(url+suf))
      """ if it is the last format to be tried"""
      if j==len(suffix)-1:
        not_found.append(book_id)
        print("\rBook_id %d not found."%book_id)  
    else:
      logger.error("URL:%s, status_code: %d"%(url+suf,r.status_code))
      print("\rURL:%s, status_code: %d"%(url+suf,r.status_code))
print("\r========================\nDOWNLOAD FINISHED.\n%d books downloaded, %d books already available, %d not found"
      %(downloaded,already_available,len(not_found)))
