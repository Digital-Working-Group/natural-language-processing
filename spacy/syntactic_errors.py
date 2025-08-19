
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