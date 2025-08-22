import spacy
from pathlib import Path
from collections import defaultdict

def most_frequent_word(nlp, file_path):
    """
    Takes in a natural language processor and file path
    Returns most frequent word and its number of occurrences
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    words_and_occurrences = defaultdict(int)
    for token in doc:
        if token.is_alpha and not token.is_stop:
            word = token.text.lower()
            words_and_occurrences[word] += 1
    most_common_count, most_common_word = max((v, k) for k, v in words_and_occurrences.items())
    return most_common_word, most_common_count

def repeating_unique_word_ratio(nlp, file_path):
    """
    Takes in a natural language processor and file path and per word amount
    calculates (number of repeating words) / (number of unique words)
    Returns the ratio of repeating to unique words
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    words_and_counts = {}
    repeating_words = 0
    for token in doc:
        if token.is_alpha:
            if token.text not in words_and_counts:
                words_and_counts[token.text] = 1
            else:
                words_and_counts[token.text] += 1
                repeating_words += 1
    return repeating_words / len(words_and_counts)

def total_consecutive_words(nlp, file_path):
    """
    Takes in a natural language processor and file path
    Counts the number of consecutive repeating words
    returns the count of consecutive repeating words.
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    word_list = []
    consecutive_words = 0
    for token in doc:
        if token.is_alpha:
            word_list.append(token.text)
    for i, word in enumerate(word_list):
        next_word = word_list[i + 1] if i != len(word_list) - 1 else None
        if word == next_word:
            consecutive_words += 1
    return consecutive_words