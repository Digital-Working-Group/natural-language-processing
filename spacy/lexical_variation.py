from pathlib import Path


def windowed_text_token_ratio(nlp, file_path, window_size=20):
    """
    Takes in a natural language processor and file path and window size (default is 20)
    Calculates type/token ratio for a fixed-length window, and averages type/token ratios from all windows.
    Returns moving average text token ratio
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    word_list = []
    text_token_ratio_per_window = []
    for token in doc:
        if token.is_alpha:
            word_list.append(token.text.lower())
    for i in range(len(word_list) - window_size + 1):
        words_in_window = word_list[i: i + window_size]
        unique_words = set(words_in_window)
        text_token_ratio_per_window.append(len(unique_words) / window_size)
    return sum(text_token_ratio_per_window) / len(text_token_ratio_per_window)

def number_of_unique_tokens(nlp, file_path):
    """
    Takes in a spacy doc and returns the number of unique tokens
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    unique_tokens = set()
    for token in doc:
        unique_tokens.add(token.text)
    return len(unique_tokens)

def number_of_unique_lemmas(nlp, file_path):
    """
    Takes in a spacy doc and returns the number of unique tokens
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    unique_lemmas = set()
    for token in doc:
        unique_lemmas.add(token.lemma_)
    return len(unique_lemmas)

