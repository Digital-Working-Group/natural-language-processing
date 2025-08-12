import spacy
import claucy
import math
import pathlib
from spellchecker import SpellChecker
import itertools
from spacy import displacy
import pandas as pd
from collections import defaultdict


def data_to_df(nlp, file_path="test.txt"):
    '''Takes in a natural language processor and a file path, and returns a pandas dataframe with token attributes '''

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    doc = nlp(text)
    d = []
    for token in doc:
        d.append({
            "TEXT": token.text,
            "LEMMA": token.lemma_,
            "POS": token.pos_,
            "TAG": token.tag_,
            "DEP": token.dep_,
            "SHAPE": token.shape_,
            "ALPHA": token.is_alpha,
            "STOP": token.is_stop
        })
    return pd.DataFrame(d)


def tag_ratio(tag="POS", amount=100):
    ''' Takes in spacy doc object, desired tag, and per word amount(default is 100)
        Returns a dictionary with pos tags and on average how many are present per 100 words'''

    df = data_to_df(nlp=spacy.load('en_core_web_lg'))
    print(df.head())
    tag_counts = defaultdict(int)
    total = 0
    #counts tags
    for t in df[tag]:
        tag_counts[t] += 1
        total += 1
    #calculate proportion
    for tag_label in tag_counts:
        tag_counts[tag_label] = tag_counts[tag_label] / total * amount

    return tag_counts


# Open the file in read mode
file_path = 'sample.txt'


with open("test.txt", 'r', encoding="utf-8") as file:
    # Read the content of the file
    file_lines = file.readlines()
    for line in file_lines:
        print(line.strip())



print(tag_ratio(tag="POS", amount=100))










