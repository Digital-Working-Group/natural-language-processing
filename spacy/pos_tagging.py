import statistics
from pathlib import Path
import spacy
from collections import defaultdict
import pandas as pd

def data_to_df(nlp, file_path):
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

def pos_tag_counts(nlp, file_path, tag, amount=100):
    """
    Takes in a natural language processor, filepath, tag("POS", or " TAG"), and per word amount
    Returns a dictionary containing all tags and on average how many times they appear per 100 words
    """
    df = data_to_df(nlp, file_path)
    tag_counts = defaultdict(int)
    for t in df[tag]:
        tag_counts[t] += 1
    total = sum(tag_counts.values())
    for tag_label in tag_counts:
        tag_counts[tag_label] = (tag_counts[tag_label] / total) * amount
    return tag_counts

def tag_ratio(nlp, file_path, amount=100):
    """
    Uses pos_tag_counts() to return a dictionary containing all parts-of-speech and Penn Treebank tags and on average how many times they appear per 100 words
    """
    return {"POS": pos_tag_counts(nlp, file_path, tag="POS", amount=amount), "TAG": pos_tag_counts(nlp, file_path, tag="TAG", amount=amount)}

def ratio_of_pos(nlp, file_path, **kwargs):
    """
    Returns the ratio of specified part of speech to total words
    """
    parts_of_speech = kwargs["parts_of_speech"]
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    total_words = 0
    total_nouns = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ in parts_of_speech:
                total_nouns += 1
    return total_nouns / total_words

def ratio_of_nouns(nlp, file_path):
    """
    Uses ratio_of_pos() with specified kwargs to calculate and return the ration of nouns to total words
    Information about the parts-of-speech tags can be found in spacy_pos_tags_explained.md
    """
    kwargs = {"parts_of_speech": ["NOUN", "PROPN"]}
    return ratio_of_pos(nlp, file_path, **kwargs)

def ratio_of_pronouns(nlp, file_path):
    """
    Uses ratio_of_pos() with specified kwargs to calculate and return the ration of pronouns to total words
    Information about the parts-of-speech tags can be found in spacy_pos_tags_explained.md
    """
    kwargs = {"parts_of_speech": ["PRON"]}
    return ratio_of_pos(nlp, file_path, **kwargs)

def ratio_of_conjunctions(nlp, file_path):
    """
    Uses ratio_of_pos() with specified kwargs to calculate and return the ration of conjunctions to total words
    Information about the parts-of-speech tags can be found in spacy_pos_tags_explained.md
    """
    kwargs = {"parts_of_speech": ["CONJ", "SCONJ"]}
    return ratio_of_pos(nlp, file_path, **kwargs)

def stats_proportion_part_of_speech(nlp, file_path, **kwargs):
    """
    Takes in a natural language processor, file path, and set of kwargs
    Calculates the ratio of specified part-of-speech to total words in a sentence for all sentences
    Returns the average, minimum, maximum, and standard deviation across all sentences
    """
    tag = kwargs.get("tag")
    dep = kwargs.get("dep")

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    pos_to_word_ratios = []

    for sentence in doc.sents:
        number_of_words = 0
        number_of_pos = 0
        for token in sentence:
            if token.is_alpha:
                number_of_words += 1
                if (tag and token.pos_ == tag) or (dep and token.dep_ == dep):
                    number_of_pos += 1
        pos_to_word_ratios.append(number_of_pos / number_of_words)

    return {
        "mean": sum(pos_to_word_ratios) / len(pos_to_word_ratios),
        "max": max(pos_to_word_ratios),
        "min": min(pos_to_word_ratios),
        "std": statistics.stdev(pos_to_word_ratios)
    }

def stats_proportion_coordinators(nlp, file_path):
    """
    Takes in a natural language processor and file path
    Uses stats_proportion_part_of_speech to determine mean, min, max, and standard deviation of the proportion of coordinators in a sentence
    """
    kwargs = {"tag": "CCONJ"}
    return stats_proportion_part_of_speech(nlp, file_path, **kwargs)

def stats_proportion_auxiliaries(nlp, file_path):
    """
    Takes in a natural language processor and file path
    Uses stats_proportion_part_of_speech to determine mean, min, max, and standard deviation of the proportion of auxiliaries in a sentence
    """
    kwargs = {"tag": "AUX"}
    return stats_proportion_part_of_speech(nlp, file_path, **kwargs)

def stats_proportion_adjectives(nlp, file_path):
    """
    Takes in a natural language processor and file path
    Uses stats_proportion_part_of_speech to determine mean, min, max, and standard deviation of the proportion of adjectives in a sentence
    """
    kwargs = {"tag": "ADJ"}
    return stats_proportion_part_of_speech(nlp, file_path, **kwargs)

def stats_proportion_subjects(nlp, file_path):
    """
    Takes in a natural language processor and file path
    Calculates the ratio of subjects to total words in a sentence
    Returns the average, minimum, maximum, and standard deviation across all sentences
    """
    kwargs = {"dep": "nsubj"}
    return stats_proportion_part_of_speech(nlp, file_path, **kwargs)