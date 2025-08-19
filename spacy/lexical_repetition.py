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