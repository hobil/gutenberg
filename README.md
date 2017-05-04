The folder implementation contains the function predicting book metadata as well the one displaying character interactions.

Folder implementation/books contains texts of books NOT used for training the classifiers. There are some texts not contained in PG.

The classifiers were trained on documents with the majority being from the 19th century and category Language and literature. It was trained on 1 400 classical authors, so the author label will be incorrect if the author was not in the training set.


RUNNING SCRIPTS
===============
For both inspect_text and character_interaction, spacy is needed:
>>> pip install spacy
>>> python3 -m spacy download en

For character_interaction, networkx is needed:
>>> pip install networkx



To make predictions for a given text, run:
>>> python3 inspect_text.py -f books/hamlet.txt

To save the pdf with character interactions:
>>> python3 character_interaction.py -f books/sherlock.txt output.pdf


