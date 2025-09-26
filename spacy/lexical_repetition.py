"""
lexical_repetition.py
Functions related to lexical repetition
"""
from collections import defaultdict

def word_repetition(nlp_util):
    """
    Only iterates over alphanumeric tokens.
    most_frequent_word: most frequent word
    most_frequent_ct: the number of times the most_frequent_word occurs
    repeating_words: the total number of words that appear >1 times.
    unique_words: the total number of unique words
    repeating_to_unique_ratio: repeating_words / unique_words
    consecutive_words: the total number of words that appear consecutively
    """
    word_ct = defaultdict(int)
    for token in nlp_util.doc:
        if token.is_alpha:
            word_ct[token.text.lower()] += 1
    most_frequent_ct, most_frequent_word = max((v, k) for k, v in word_ct.items())
    repeating_words = sum((v for v in word_ct.values() if v > 1))
    unique_words = len(word_ct)
    repeating_to_unique_ratio = repeating_words / unique_words
    consecutive_words = 0
    word_list = list(word_ct.keys())
    for idx, word in enumerate(word_list):
        if idx == unique_words - 1: ## reached the end of word_list
            break
        if word == word_list[idx + 1]:
            consecutive_words += 1
    function = 'word_repetition'
    parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath, 'function': function}
    data = {'most_frequent_word': most_frequent_word, 'most_frequent_ct': most_frequent_ct,
        'repeating_words': repeating_words, 'unique_words': unique_words,
        'repeating_to_unique_ratio': repeating_to_unique_ratio,
        'consecutive_words': consecutive_words}
    final_data = {'parameters': parameters, 'data': data}
    nlp_util.write_json_model(function, final_data)
