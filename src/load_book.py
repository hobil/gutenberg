import os
import re
from gutenberg import cleanup

FORMATS=['.txt','-0.txt','-8.txt','','-0','-8']

def load_book(book_id,books_folder='../res/txts',preprocessed_folder='../res/preprocessed'):
  """
  Loads book with the given id.
  Returns bookid and book's text without footers and headers.
  The path must end with '/'
  """
  
  if not os.path.exists(preprocessed_folder):
    os.mkdir(preprocessed_folder)
  
  preprocessed_book_filename=os.path.join(preprocessed_folder,str(book_id))
  if os.path.exists(preprocessed_book_filename):
    return open(preprocessed_book_filename,encoding='utf-8',errors='ignore').read()
  else:
    def open_book_file(book_id):
      """
      Tries to find book text in in one of the specified formats 
      in the given folder path.
      """
      # book exists in the root folder
      for f in FORMATS:
        filename=os.path.join(books_folder,book_id+f)
        if os.path.exists(filename):
          # it is not a directory
          if os.path.isfile(filename):
            return open(filename,encoding='utf-8',errors='ignore').read()

      # book is in the folder root/bookid
      for f in FORMATS:
        filename=os.path.join(books_folder,book_id,book_id+f)
        if os.path.exists(filename):
          # it is not a directory
          if os.path.isfile(filename):
            return open(filename,encoding='utf-8',errors='ignore').read()

    #=========end of open_book_file=========================#
      
    # start of load_book function
    text=open_book_file(str(book_id))
    if text==None:
      print("Book with id %d not found"%book_id)
      return
    # Removes header and footer from the text.
    cleaned_book_text=(cleanup.strip_headers(text))
    
    filtered_text=filter_doc(cleaned_book_text)
    
    #save the preprocessed_version
    with open(preprocessed_book_filename,'w',encoding='utf-8',errors='ignore') as f:
      f.write(filtered_text)
        
    return filtered_text
  
def filter_doc(doc_text):
  """
  Does minor preprocessing on the text.
  """
  # remove stars
  filter_regex=re.compile(r"[_*]")
  doc=filter_regex.sub("",doc_text)
  # substitute quotation marks
  double_quot_regex=re.compile(r"[“”]")
  single_quot_regex=re.compile(r"[’‘]")
  doc=double_quot_regex.sub('"',doc)
  doc=single_quot_regex.sub("'",doc)
  # substitute new lines inside the text for spaces
  # these new lines are usually caused by formatting texts to fit in 80 columns 
  newline_quot_regex=re.compile(r"(\S)\n(\S)")
  doc=newline_quot_regex.sub(r"\1 \2",doc)
  # remove illustration tag
  #illustration_regex=re.compile(r"\[Illustration.*]")
  #doc=illustration_regex.sub("",doc)
  return doc
  

def load_and_filter_document(book_id):

  doc=load_book(book_id)
  if doc==None:
    return
  return filter_doc(doc)

