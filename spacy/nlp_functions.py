from collections import defaultdict
import pandas as pd
import itertools
import math
import statistics
import spacy
from pathlib import Path

def frequency_nonwords(nlp=None, file_path=None, amount=100):
    """
    Takes in a natural language processor and file path and per word amount(default is 100)
    Calculates frequency of non-words using dataset of english words
    Outputs on average how many non-words are present per word amount
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    total_words = 0
    total_nonwords = 0
    #english words for comparison
    word_set = set()
    with open("words_alpha.txt", "r") as f:
        for line in f:
            word_set.add(line.strip())

    #check if word exists in dataset
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.text.lower() not in word_set and token.lemma_.lower() not in word_set:
                total_nonwords += 1

    ratio_nonwords = total_nonwords / total_words * amount
    return ratio_nonwords


def length_of_sentences(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Calculates average length of sentences
    Returns the average length of sentences
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    num_sentences = 0
    words_in_sentence = []
    #count number of sentences and words in sentence
    for sentence in doc.sents:
        num_words = 0
        num_sentences += 1
        for token in sentence:
            if token.is_alpha:
                num_words += 1
        words_in_sentence.append(num_words)
    avg_words_per_sentence = sum(words_in_sentence) / len(words_in_sentence)
    return avg_words_per_sentence

#Occurances of the most frequent token in the text
def occurrences_of_most_frequent(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Finds most frequent word and counts its occurrences
    Returns most frequent word and occurrences
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    words_and_occurrences = {}
    most_common = 0
    most_common_word = None
    #find word occurrences excluding common words
    for token in doc:
        if token.is_alpha and not token.is_stop:
            word = token.text.lower()
            if word not in words_and_occurrences:
                words_and_occurrences[word] = 1
            else:
                words_and_occurrences[word] += 1

    for word in words_and_occurrences:
        if words_and_occurrences[word] > most_common:
            most_common = words_and_occurrences[word]
            most_common_word = word

    return most_common_word, most_common


def moving_average_text_token_ratio(nlp=None, file_path=None, window_size=20):
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

def term_frequency(nlp=None, file_path=None, term=None):
    """
    Takes in a natural language processor and file path and target string
    Returns frequency of target string in document
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    term_appearances = 0
    total_words = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
        if token.text.lower() == term:
            term_appearances += 1
    return term_appearances / total_words

def inverse_document_frequency(document_list, term=None):
    """
    Takes in a document and target string
    Calculates inverse document frequency by taking log of (number of documents) / (number of documents containing term)
    returns inverse document frequency value
    """

    number_of_documents = len(document_list)
    documents_containing_term = 0

    for document in document_list:
        term_freq_doc = term_frequency(document, term)
        if term_freq_doc > 0:
            documents_containing_term += 1

    return math.log10(number_of_documents / documents_containing_term)

def tf_idf(term=None, document_list=None):
    """
    Takes in a document list, spacy doc object, and target string
    Calculates TF-IDF by multiplying TF by IDF
    returns TF-IDF value
    """

    return term_frequency() * inverse_document_frequency(document_list, term)

def repeating_unique_word_ratio(nlp=None, file_path=None, amount=100):
    """
    Takes in a natural language processor and file path and per word amount
    calculates (number of repeating words) / (number of unique words)
    Returns number of repeating words per word amount
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    word_list = []
    words_and_counts = {}
    repeating_words = 0
    for token in doc:
        if token.is_alpha:
            word_list.append(token.text)
            if token.text not in words_and_counts:
                words_and_counts[token.text] = 1
            else:
                words_and_counts[token.text] += 1
                repeating_words += 1

    unique_words = set(word_list)
    return repeating_words / len(unique_words) * amount

def incorrectly_followed_articles(nlp=None, file_path=None):
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
            following_token = doc[i+1]
            if following_token.pos_ not in ["ADJ", "NOUN", "PROPN"]:
                count += 1
    return count

def number_of_unique_tokens(nlp=None, file_path=None):
    """
    Takes in a spacy doc and returns the number of unique tokens
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    unique_tokens = set()
    for token in doc:
        unique_tokens.add(token.text)
    return len(unique_tokens)

def number_of_unique_lemmas(nlp=None, file_path=None):
    """
    Takes in a spacy doc and returns the number of unique lemmas
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    lemmas = set()
    for token in doc:
        lemmas.add(token.lemma_)
    return len(lemmas)

def ratio_of_nouns(nlp=None, file_path=None):
    """
    Takes in a spacy doc and returns the ratio of nouns to total words
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    total_words = 0
    total_nouns = 0
    for token in doc:
        if token.is_alpha:
            print(token)
            total_words += 1
            if token.pos_== "NOUN":
                total_nouns += 1
    return total_nouns / total_words



def avg_wh_words(nlp=None, file_path=None, amount=100):
    """
    Takes in a spacy doc and per word amount
    calculates number of wh words per word amount
    returns average number of wh-words per word amount
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    total_wh_words = 0
    total_words = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            word = token.text.lower()
            if word[0:2] == "wh":
                total_wh_words += 1
    return total_wh_words / total_words * amount

def avg_num_nonwords(nlp=None, file_path=None, amount=100):
    """
    Takes in doc object and word amount
    Counts number of nonwords by checking if words are in spacys vocabulary
    Returns average number of nonwords per word amount
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    nonwords = 0
    total_words = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.is_oov:
                nonwords += 1
    return nonwords / total_words * amount

def mean_similarity_of_sentences(nlp=None, file_path=None):
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

def tree_height(node):
    """
    Returns the max height of a tree given a node
    """
    if node is None:
        return 0
    else:
        children = list(node.lefts) + list(node.rights)
        if not children:
            return 1
        else:
            return max(tree_height(child) for child in children) + 1

def avg_dependency_tree_height(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Uses tree_height() to calculate max height of dependency trees
    Returns average height of all dependency trees
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    tree_depths = []
    max_depth = 0
    for sentence in doc.sents:
        for token in sentence:
            if token.dep_ == "ROOT":
                max_depth = tree_height(token)
        tree_depths.append(max_depth)
    return sum(tree_depths) / len(tree_depths)

def max_dependency_tree_height(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Uses tree_height() to calculate max height of dependency trees
    Returns average height of all dependency trees
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    tree_depths = []
    max_depth = 0
    for sentence in doc.sents:
        for token in sentence:
            if token.dep_ == "ROOT":
                max_depth = tree_height(token)
        tree_depths.append(max_depth)
    return max(tree_depths)


def stats_similarity_of_words(nlp=None, file_path=None, window_size=3, stat=None):
    """
    Takes in a natural language processor and file path, window size(default 3)
    Computes standard deviation, mean or max (depending on stat param) of similarity between words in each window
    Returns statistic of similarity score across all windows (max, mean, std)
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    word_list = []
    similarity_scores = []
    # make word list
    for token in doc:
        if token.is_alpha:
            word_list.append(token.text)
    # get list of words in window
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

    if stat.lower() == "stdev":
        stat_similarity = statistics.stdev(similarity_scores)
    elif stat.lower() == "mean":
        stat_similarity = sum(similarity_scores) / len(similarity_scores)
    elif stat.lower() == "max":
        stat_similarity = max(similarity_scores)
    else:
        stat_similarity = None

    return stat_similarity


def std_similarity_of_words(nlp=None, file_path=None, window_size=3):
    """
    Takes in a natural language processor, file path and window size(default 3)
    utilizes stats_similarity_of_words() to calculate standard deviation of word similarity across moving windows, returns this value
    """
    return stats_similarity_of_words(nlp=nlp, file_path=file_path, window_size=window_size, stat="stdev")


def mean_similarity_of_words(nlp=None, file_path=None, window_size=3):
    """
    Takes in a natural language processor, file path and window size(default 3)
    utilizes stats_similarity_of_words() to calculate mean word similarity across moving windows, returns this value
    """
    return stats_similarity_of_words(nlp=nlp, file_path=file_path, window_size=window_size, stat="mean")


def max_similarity_of_words(nlp=None, file_path=None, window_size=3):
    """
    Takes in a natural language processor, file path and window size(default 3)
    utilizes stats_similarity_of_words() to calculate maximum word similarity across moving windows, returns this value
    """

    return stats_similarity_of_words(nlp=nlp, file_path=file_path, window_size=window_size, stat="max")


def ratio_of_pronouns(nlp=None, file_path=None):
    """
    Takes in a spacy doc and returns the ratio of nouns to total words
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    total_words = 0
    total_pronouns = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_== "PRON":
                total_pronouns += 1
    return total_pronouns / total_words

def ratio_of_conjunctions(nlp=None, file_path=None):
    """
    Takes in a spacy doc and returns the ratio of nouns to total words
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    total_words = 0
    total_conjunctions = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_== "CCONJ" or token.pos_ == "SCONJ":
                total_conjunctions += 1
    return total_conjunctions / total_words

def stats_proportion_part_of_speech(nlp=None, file_path=None, pos_index=None):
    """
    Takes in a natural language processor and file path
    Calculates the ratio of coordinators to total words in a sentence
    Returns the average, minimum, maximum, and standard deviation across all sentences
    """
    pos_list = ["adjectives", "auxiliaries", "coordinators"]
    tag_list = ["ADJ", "AUX", "CCONJ"]

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    pos_to_word_ratios = []
    stats_num_pos = {}
    for sentence in doc.sents:
        number_of_words = 0
        number_of_pos = 0
        for token in sentence:
            if token.is_alpha:
                number_of_words += 1
                if token.pos_ == tag_list[pos_index]:
                    number_of_pos += 1
        pos_to_word_ratios.append(number_of_pos / number_of_words)

    stats_num_pos["mean"] = sum(pos_to_word_ratios) / len(pos_to_word_ratios)
    stats_num_pos["maximum"] = max(pos_to_word_ratios)
    stats_num_pos["minimum"] = min(pos_to_word_ratios)
    stats_num_pos["standard deviation"] = statistics.stdev(pos_to_word_ratios)
    return stats_num_pos

def stats_proportion_coordinators(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Uses stats_proportion_part_of_speech to determine mean, min, max, and standard deviation of the proportion of coordinators in a sentence
    """
    return stats_proportion_part_of_speech(nlp=nlp, file_path=file_path, pos_index=2)

def stats_proportion_auxiliaries(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Uses stats_proportion_part_of_speech to determine mean, min, max, and standard deviation of the proportion of auxiliaries in a sentence
    """
    return stats_proportion_part_of_speech(nlp=nlp, file_path=file_path, pos_index=1)

def stats_proportion_adjectives(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Uses stats_proportion_part_of_speech to determine mean, min, max, and standard deviation of the proportion of adjectives in a sentence
    """
    return stats_proportion_part_of_speech(nlp=nlp, file_path=file_path, pos_index=0)

def stats_proportion_subjects(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Calculates the ratio of subjects to total words in a sentence
    Returns the average, minimum, maximum, and standard deviation across all sentences
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    subjects_to_word_ratios = []
    stats_num_subjects = {}
    for sentence in doc.sents:
        number_of_words = 0
        number_of_subjects = 0
        for token in sentence:
            if token.is_alpha:
                number_of_words += 1
                if token.dep_ == "nsubj":
                    number_of_subjects += 1
        subjects_to_word_ratios.append(number_of_subjects / number_of_words)

    stats_num_subjects["mean"] = sum(subjects_to_word_ratios) / len(subjects_to_word_ratios)
    stats_num_subjects["maximum"] = max(subjects_to_word_ratios)
    stats_num_subjects["minimum"] = min(subjects_to_word_ratios)
    stats_num_subjects["standard deviation"] = statistics.stdev(subjects_to_word_ratios)
    return stats_num_subjects

def count_num_sentences_without_verbs(nlp=None, file_path=None):
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

def total_consecutive_words(nlp=None, file_path=None):
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