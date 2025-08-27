import statistics
import spacy
from pathlib import Path
import itertools

def stats_similarity_of_words(nlp, file_path, window_size=3):
    """
    Takes in a natural language processor and file path, window size(default 3)
    Returns dictionary containing mean,min, max, and standard deviation of word similarity across all windows
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    word_list = []
    similarity_scores = []
    for token in doc:
        if token.is_alpha:
            word_list.append(token.text)
    for i in range(len(word_list) - window_size + 1):
        similarity_scores_of_window = []
        words_in_window = word_list[i: i + window_size]
        for combination in itertools.combinations(words_in_window, 2):
            word1 = str(combination[0])
            word2 = str(combination[1])
            doc1 = nlp(word1)
            doc2 = nlp(word2)
            similarity_score = round(doc1.similarity(doc2), 2)
            similarity_scores_of_window.append(similarity_score)

        avg_similarity_of_window = sum(similarity_scores_of_window) / len(similarity_scores_of_window)
        similarity_scores.append(avg_similarity_of_window)

    return {
        "mean": sum(similarity_scores) / len(similarity_scores),
        "max": max(similarity_scores),
        "min": min(similarity_scores),
        "std": statistics.stdev(similarity_scores)
    }

def mean_similarity_of_sentences(nlp, file_path):
    """
    Takes in a natural language processor and file path
    calculates mean similarity of all combinations of sentences
    Returns mean similarity of all sentences
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    sentence_list = list(doc.sents)
    similarity_scores = []
    for comb in itertools.combinations(sentence_list, 2):
        sentence_one = str(comb[0])
        sentence_two = str(comb[1])
        doc_one = nlp(sentence_one)
        doc_two = nlp(sentence_two)
        similarity_score = round(doc_one.similarity(doc_two), 2)
        similarity_scores.append(similarity_score)
    return sum(similarity_scores) / len(similarity_scores)

