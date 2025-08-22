
import spacy
from pathlib import Path

def nonword_frequency(nlp, file_path, dataset_fp, amount=100):
    """
    Takes in a natural language processor and file path, dataset of words and per word amount(default is 100)
    Dataset used in examples comes from kaggle, it is text file containing over 466k English words
    Calculates frequency of non-words using dataset
    Outputs on average how many non-words are present per word amount
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    total_words = 0
    total_nonwords = 0
    word_set = set()
    with open(dataset_fp, "r") as f:
        for line in f:
            word_set.add(line.strip())
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.text.lower() not in word_set and token.lemma_.lower() not in word_set:
                total_nonwords += 1
    ratio_nonwords = total_nonwords / total_words * amount
    return ratio_nonwords

def incorrectly_followed_articles(nlp, file_path):
    """
    Takes in a natural language processor and file path
    counts number of articles not followed by a noun, proper noun, or adjective
    returns the count
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    articles = ["a", "an", "the"]
    count = 0
    for i, token in enumerate(doc):
        if token.text.lower() in articles:
            try:
                following_token = doc[i+1]
                if following_token.pos_ not in ["ADJ", "NOUN", "PROPN"]:
                    count += 1
            except IndexError:
                count += 1
    return count

def count_num_sentences_without_verbs(nlp, file_path):
    """
    Takes in a natural language processor and file path
    Counts the number of sentences without verbs.
    Returns the count of sentences without verbs.
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    count = 0
    for sentence in doc.sents:
        count_verbs =  0
        for token in sentence:
            if token.pos_ == "VERB":
                count_verbs += 1
        if count_verbs < 1:
            count += 1
    return count