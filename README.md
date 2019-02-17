The folder implementation contains the function predicting book metadata as well the one displaying character interactions.

Folder implementation/books contains texts of books NOT used for training the classifiers. There are some texts not contained in PG.

The classifiers were trained on documents with the majority being from the 19th century and category Language and literature. It was trained on 1 400 classical authors, so the author label will be incorrect if the author was not in the training set.


RUNNING SCRIPTS
===============
The code is tested for python 3.6. First, install required modules:

`pip install -r requirements.txt`

Then download English corpus for `spacy`:

`python -m spacy download en`

Download stopwords and punkt tokenizer for `nltk`:

```
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
```

To make predictions for a given text, run:

`python inspect_text.py -f books/hamlet.txt`

To save the pdf with character interactions:

`python character_interaction.py -f books/sherlock.txt output.pdf`
