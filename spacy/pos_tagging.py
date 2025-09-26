"""
pos_tagging.py
Functions related to parts of speech tagging.
"""
import statistics
from collections import defaultdict
import pandas as pd
import spacy

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

def pos_tag_ratio(nlp_util, tag_list, amount=100):
    """
    get the tag ratio for each tag in tag_list
    """
    function = 'pos_tag_ratio'
    parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath, 'tag_list': tag_list,
        'amount': amount, 'function': function}
    doc_df = data_to_df(nlp_util.doc)
    all_tag_data = []
    for tag in tag_list:
        tag_data, total = get_tag_data_and_total(doc_df, tag, amount)
        all_tag_data.append({'tag': tag, 'total_tokens': total, 'tag_data': tag_data})
    final_data = {'parameters': parameters, 'data': all_tag_data}
    nlp_util.write_json_model(function, final_data)

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

def alpha_pos_ratio(nlp_util, **kwargs):
    """
    Examines only alphanumeric (is_alpha) characters
    Returns the ratio of specific part(s) of speech to total words
    Details about the spaCy POS tags  in spacy_pos_tags_explained.md

    model: spaCy model to load
    filepath: text file to process
    pos_to_list: POS string name to the list of tags associated with the desired POS
        {'nouns': ['NOUN', 'PROPN']}
        {'pronouns': ['PRON']}
        {'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}

    Writes an output JSON for each key, value pair in pos_to_list.
    """
    pos_to_list = kwargs['pos_to_list']
    function = 'alpha_pos_ratio'
    pos_to_ct, total_tokens = get_alpha_pos_ct_and_total(nlp_util.doc, pos_to_list)
    for pos, pos_ct in pos_to_ct.items():
        parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath,
            'pos_list': pos_to_list[pos], 'function': function}
        final_data = {'parameters': parameters, 'data': {'pos_ratio': pos_ct / total_tokens}}
        nlp_util.write_json_model(function, final_data, ext=pos)

def get_pos_to_ratios(doc, pos_to_list):
    """
    get pos_to_ratios for alpha_pos_ratio_sentences()
    """
    pos_to_ratios = defaultdict(list)
    for sentence in doc.sents:
        pos_to_ct, total_tokens = get_alpha_pos_ct_and_total(sentence, pos_to_list)
        for pos, pos_ct in pos_to_ct.items():
            pos_to_ratios[pos].append(pos_ct / total_tokens)
    return pos_to_ratios

def calc_sent_stats(list_of_ratios):
    """
    get total, mean, max, mind, std of ratios for alpha_pos_ratio_sentences()
    """
    sent_total = len(list_of_ratios)
    sent_mean = sum(list_of_ratios) / sent_total
    sent_max = max(list_of_ratios)
    sent_min = min(list_of_ratios)
    sent_std = statistics.stdev(list_of_ratios) if sent_total >= 2 else None
    return {'sent_mean': sent_mean, 'sent_max': sent_max, 'sent_min': sent_min,
            'sent_std': sent_std, 'sent_total': sent_total}

def alpha_pos_ratio_sentences(nlp_util, **kwargs):
    """
    Examines only alphanumeric (is_alpha) characters
    Calculates the ratio of specified part-of-speech to total words per sentence
    Returns the average, minimum, maximum, and standard deviation of the ratio across all sentences

    model: spaCy model to load
    filepath: text file to process
    pos_to_list: POS string name to the list of tags associated with the desired POS
        {'nouns': ['NOUN', 'PROPN']}
        {'pronouns': ['PRON']}
        {'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}
    """
    pos_to_list = kwargs['pos_to_list']
    function = 'alpha_pos_ratio_sentences'
    pos_to_ratios = get_pos_to_ratios(nlp_util.doc, pos_to_list)
    for pos, list_of_ratios in pos_to_ratios.items():
        parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath,
            'pos_list': pos_to_list[pos], 'function': function}
        data = calc_sent_stats(list_of_ratios)
        final_data = {'parameters': parameters, 'data': data}
        nlp_util.write_json_model(function, final_data, ext=pos)
