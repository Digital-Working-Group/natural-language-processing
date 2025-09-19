"""
pos_tagging.py
Functions related to parts of speech tagging.
"""
from collections import defaultdict
from pathlib import Path
import pandas as pd
import spacy
import utility as util

def data_to_df(doc):
    """
    Takes in a spacy doc and returns a pandas dataframe with token attributes

    Text: The original word text.
    Lemma: The base form of the word.
    POS: The simple UPOS part-of-speech tag.
    Tag: The detailed part-of-speech tag.
    Dep: Syntactic dependency, i.e. the relation between tokens.
    Shape: The word shape – capitalization, punctuation, digits.
    is alpha: Is the token an alpha character?
    is stop: Is the token part of a stop list, i.e. the most common words of the language?

    """
    token_list = []
    for token in doc:
        token_list.append({
            "TEXT": token.text,
            "LEMMA": token.lemma_,
            "POS": token.pos_,
            "TAG": token.tag_,
            "DEP": token.dep_,
            "SHAPE": token.shape_,
            "ALPHA": token.is_alpha,
            "STOP": token.is_stop
        })
    return pd.DataFrame(token_list)

def get_pos_tag_ratio(doc_df, tag, amount=100):
    """
    parameters::
        doc_df: pandas DataFrame from data_to_df()
        tag: Desired tag to calculate counts for (e.g., 'POS' or 'TAG')
        amount: Count how many times each tag value occurs per <amount> words
    return:
        tag_counts: {tag_label: (total tag count, tag ratio)}
            tag_ratio = total tag count / total tokens * amount
        total: total tokens
    """
    tag_counts = defaultdict(int)
    for tag_label in doc_df[tag]:
        tag_counts[tag_label] += 1
    total = sum(tag_counts.values())
    for tag_label, tag_ct in tag_counts.items():
        tag_counts[tag_label] = (tag_ct, (tag_ct / total) * amount)
    return tag_counts, total

def get_tag_data_and_total(doc_df, tag, amount):
    """
    get tag_data and total
    """
    tag_data = []
    tag_counts, total = get_pos_tag_ratio(doc_df, tag, amount=amount)
    for tag_label, (tag_ct, tag_ratio) in tag_counts.items():
        tag_data.append({'tag_label': tag_label, 'tag_ct': tag_ct,
            'tag_ratio': tag_ratio, 'spacy.explain': spacy.explain(tag_label)})
    return tag_data, total

def pos_tag_ratio(model, filepath, tag_list, amount=100):
    """
    get the tag ratio for each tag in tag_list
    """
    nlp = spacy.load(model)
    function = 'pos_tag_ratio'
    parameters = {'model': model, 'filepath': filepath, 'tag_list': tag_list,
        'amount': amount, 'function': function}
    path_filepath = Path(filepath)
    doc_df = data_to_df(nlp(path_filepath.read_text(encoding='utf-8')))
    all_tag_data = []
    for tag in tag_list:
        tag_data, total = get_tag_data_and_total(doc_df, tag, amount)
        all_tag_data.append({'tag': tag, 'total_tokens': total, 'tag_data': tag_data})
    final_data = {'parameters': parameters, 'data': all_tag_data}
    util.write_json_model(path_filepath, function, model, final_data)

def get_alpha_pos_ct_and_total(doc, pos_to_list):
    """
    get post_ct, total for alpha_pos_ratio()
    """
    total_tokens = 0
    pos_to_ct = defaultdict(int)
    for token in doc:
        if token.is_alpha:
            total_tokens += 1
            for pos, pos_list in pos_to_list.items():
                if token.pos_ in pos_list:
                    pos_to_ct[pos] += 1
    return pos_to_ct, total_tokens

def alpha_pos_ratio(model, filepath, **kwargs):
    """
    Examines only alphanumeric (is_alpha) characters
    Returns the ratio of specific part(s) of speech to total words
    Information about the parts-of-speech tags can be found in spacy_pos_tags_explained.md

    model: spaCy model to load
    filepath: text file to process
    pos_to_list: POS string name to the list of tags associated with the desired POS
        {'nouns': ['NOUN', 'PROPN']}
        {'pronouns': ['PRON']}
        {'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}

    Writes an output JSON for each key, value pair in pos_to_list.
    """
    nlp = spacy.load(model)
    path_filepath = Path(filepath)
    doc = nlp(path_filepath.read_text(encoding='utf-8'))
    pos_to_list = kwargs['pos_to_list']
    function = 'alpha_pos_ratio'
    pos_to_ct, total_tokens = get_alpha_pos_ct_and_total(doc, pos_to_list)
    for pos, pos_ct in pos_to_ct.items():
        parameters = {'model': model, 'filepath': filepath, 'pos_list': pos_to_list[pos],
            'function': function}
        final_data = {'parameters': parameters, 'data': {'pos_ratio': pos_ct / total_tokens}}
        util.write_json_model(path_filepath, function, model, final_data, ext=pos)

# def stats_proportion_part_of_speech(nlp, file_path, **kwargs):
#     """
#     Takes in a natural language processor, file path, and set of kwargs
#     Calculates the ratio of specified part-of-speech to total words in a sentence for all sentences
#     Returns the average, minimum, maximum, and standard deviation across all sentences
#     """
#     tag = kwargs.get("tag")
#     dep = kwargs.get("dep")

#     doc = nlp(Path(file_path).read_text(encoding='utf-8'))

#     pos_to_word_ratios = []

#     for sentence in doc.sents:
#         number_of_words = 0
#         number_of_pos = 0
#         for token in sentence:
#             if token.is_alpha:
#                 number_of_words += 1
#                 if (tag and token.pos_ == tag) or (dep and token.dep_ == dep):
#                     number_of_pos += 1
#         pos_to_word_ratios.append(number_of_pos / number_of_words)

#     return {
#         "mean": sum(pos_to_word_ratios) / len(pos_to_word_ratios),
#         "max": max(pos_to_word_ratios),
#         "min": min(pos_to_word_ratios),
#         "std": statistics.stdev(pos_to_word_ratios)
#     }

# def stats_proportion_coordinators(nlp, file_path):
#     """
#     Takes in a natural language processor and file path
#     Uses stats_proportion_part_of_speech to determine mean, min, max, and standard deviation of the proportion of coordinators in a sentence
#     """
#     kwargs = {"tag": "CCONJ"}
#     return stats_proportion_part_of_speech(nlp, file_path, **kwargs)

# def stats_proportion_auxiliaries(nlp, file_path):
#     """
#     Takes in a natural language processor and file path
#     Uses stats_proportion_part_of_speech to determine mean, min, max, and standard deviation of the proportion of auxiliaries in a sentence
#     """
#     kwargs = {"tag": "AUX"}
#     return stats_proportion_part_of_speech(nlp, file_path, **kwargs)

# def stats_proportion_adjectives(nlp, file_path):
#     """
#     Takes in a natural language processor and file path
#     Uses stats_proportion_part_of_speech to determine mean, min, max, and standard deviation of the proportion of adjectives in a sentence
#     """
#     kwargs = {"tag": "ADJ"}
#     return stats_proportion_part_of_speech(nlp, file_path, **kwargs)

# def stats_proportion_subjects(nlp, file_path):
#     """
#     Takes in a natural language processor and file path
#     Calculates the ratio of subjects to total words in a sentence
#     Returns the average, minimum, maximum, and standard deviation across all sentences
#     """
#     kwargs = {"dep": "nsubj"}
#     return stats_proportion_part_of_speech(nlp, file_path, **kwargs)
