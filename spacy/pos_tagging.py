from pathlib import Path
import spacy
from collections import defaultdict
import pandas as pd

def data_to_df(nlp=None, file_path=None):
    """ Takes in a natural language processor and a file path, and returns a pandas dataframe with token attributes """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

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


def tag_ratio(nlp=None, file_path=None, amount=100):
    """
    Takes in a natural language processor, and per word amount(default is 100)
    Returns a dictionary with parts-of-speech, tags and on average how many are present per 100 words
    """
    df = data_to_df(nlp, file_path)
    tag_counts = defaultdict(int)
    pos_counts = defaultdict(int)
    pos_and_tag_counts = {}
    total = 0
    #counts tags
    for t in df["TAG"]:
        tag_counts[t] += 1
        total += 1
    for p in df["POS"]:
        pos_counts[p] += 1
    #calculate proportion
    for tag_label in tag_counts:
        tag_counts[tag_label] = tag_counts[tag_label] / total * amount
    for pos_label in pos_counts:
        pos_counts[pos_label] = pos_counts[pos_label] / total * amount

    #dictionary with pos and tag counts
    pos_and_tag_counts["POS"] = pos_counts
    pos_and_tag_counts["TAG"] = tag_counts

    return pos_and_tag_counts
