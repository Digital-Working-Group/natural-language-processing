from collections import defaultdict
import pandas as pd
import itertools
import math
import statistics
import spacy
from pathlib import Path

def data_to_df(nlp=None, file_path=None):
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


def tag_ratio(nlp=None, file_path=None, amount=100):
    """
    Takes in a natural language processor, and per word amount(default is 100)
    Returns a dictionary with parts-of-speech, tags and on average how many are present per 100 words
    """
    df = data_to_df(nlp, file_path)
    tag_counts = defaultdict(int)
    pos_counts = defaultdict(int)
    pos_and_tag_counts = {}
    total = 0
    #counts tags
    for t in df["TAG"]:
        tag_counts[t] += 1
        total += 1
    for p in df["POS"]:
        pos_counts[p] += 1
    #calculate proportion
    for tag_label in tag_counts:
        tag_counts[tag_label] = tag_counts[tag_label] / total * amount
    for pos_label in pos_counts:
        pos_counts[pos_label] = pos_counts[pos_label] / total * amount

    #dictionary with pos and tag counts
    pos_and_tag_counts["POS"] = pos_counts
    pos_and_tag_counts["TAG"] = tag_counts

    return pos_and_tag_counts


def num_tense_inflected_verbs(nlp=None, file_path=None, amount=100):
    """
    Takes in a natural language processor, file path, and per word amount(default is 100)
    Filters on tokens that only contain alphabetic characters (excluding punctuation or numbers), and calculates avg number of tense inflected verbs per word amount
    Tokens with the pos: "AUX" are auxiliaries, and those with the tag: "MD" are modal auxiliaries
    Returns the number of tense inflected verbs
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    total_words = 0
    present_verbs = 0
    past_verbs = 0
    modal_auxiliary = 0
    num_tiv = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            #find tense inflected verbs
            if token.pos_ == "VERB" and "Pres" in token.morph.get("Tense"):
                present_verbs += 1
            if token.pos_ == "VERB" and "Past" in token.morph.get("Tense"):
                past_verbs += 1
            if token.pos_ == "AUX" and token.tag_ == "MD":
                modal_auxiliary += 1

            num_tiv = present_verbs + modal_auxiliary + past_verbs
    return num_tiv / total_words * amount

def calculate_idea_density(nlp=None, file_path=None):
    """
    Takes in a natural language processor, and file path
    Filters on alphabetic tokens (words), calculates idea density - the number of propositions divided by the total words
    Returns each sentence and its idea density
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    idea_density_sentences = []
    for sent in doc.sents:
        count_props = 0
        count_words = 0
        for token in sent:
            if token.is_alpha:
                count_words += 1
                if token.pos_ in ["VERB", "ADJ", "ADV", "CCONJ", "SCONJ", "ADP"]:
                    count_props += 1
        idea_density = count_props/count_words
        sentence_and_idea_density = (sent.text, idea_density)
        idea_density_sentences.append(sentence_and_idea_density)

    return idea_density_sentences

def abstractness(nlp=None, file_path=None, dataset_path="datasets/dataset_for_abstractness.xlsx"):
    """
    Takes in a natural language processor, file path, and dataset path
    Calculates abstractness - inverse of concreteness value taken from dataset
    Returns the average abstractness
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    df = pd.read_excel(dataset_path)
    df['Word'] = df['Word'].str.lower()

    total_words = 0
    total_abstractness_values = 0

    for token in doc:
        if token.is_alpha:
            total_words += 1
            #find concreteness value for nouns in data
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['Word'] == word, 'Conc.M']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['Word'] == word_lemma, 'Conc.M']
                if not result.empty:
                    total_abstractness_values += 1 / result.item()
                elif not result_lemma.empty:
                    total_abstractness_values += 1 / result_lemma.item()
                else:
                    total_words -= 1  #omits words not in data from calculation
    return total_abstractness_values / total_words

def semantic_ambiguity(nlp=None, file_path=None, dataset_path="datasets/dataset_for_semantic_ambiguity.xlsx"):
    """
    Takes in a natural language processor, file path, dataset path
    Calculates semantic ambiguity using semantic diversity value from dataset
    Returns the average semantic ambiguity value for all words
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    df = pd.read_excel(dataset_path, skiprows=1)
    df['!term'] = df['!term'].str.lower()

    total_words = 0
    total_ambiguity_values = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            #find semantic diversity value for nouns
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['!term'] == word, 'SemD']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['!term'] == word_lemma, 'SemD']
                if not result.empty:
                    total_ambiguity_values += result.item()
                elif not result_lemma.empty:
                    total_ambiguity_values += result_lemma.item()
                else:
                    total_words -= 1  #omit words not in data from calculation
    return total_ambiguity_values / total_words

def word_frequency(nlp=None, file_path=None, dataset_path="datasets/dataset_for_word_frequency.xlsx"):
    """
    Takes in a natural language processor, file path, dataset path
    Calculates word frequency using log10 word frequency measure from dataset
    Returns the average word frequency value for all words
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    df = pd.read_excel(dataset_path)
    df['Word'] = df['Word'].str.lower()

    total_words = 0
    total_wf_values = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            #find word frequency value for nouns in data
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['Word'] == word, 'Lg10WF']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['Word'] == word_lemma, 'Lg10WF']

                if not result.empty:
                    total_wf_values += result.item()
                elif not result_lemma.empty:
                    total_wf_values += result_lemma.item()
                else:
                    total_words -= 1   #omit words not in data from calculation
    return total_wf_values / total_words

def word_prevalence(nlp=None, file_path=None, dataset_path="datasets/dataset_for_word_prevalence_and_familiarity.xlsx"):
    """
    Takes in a natural language processor, file path, dataset path
    Calculates word prevalence using prevalence measure from dataset
    Returns the average word prevalence value of all words per
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    df = pd.read_excel(dataset_path)
    df['Word'] = df['Word'].str.lower()

    total_words = 0
    total_wp_values = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            #find word prevalence values for nouns from data
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['Word'] == word, 'Prevalence']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['Word'] == word_lemma, 'Prevalence']

                if not result.empty:
                    total_wp_values += result.item()
                elif not result_lemma.empty:
                    total_wp_values += result_lemma.item()
                else:
                    total_words -= 1  #omit words not in data from calculation
    return total_wp_values / total_words


def word_familiarity(nlp=None, file_path=None, dataset_path="datasets/dataset_for_word_prevalence_and_familiarity.xlsx"):
    """
    Takes in a natural language processor, file path, dataset path
    Calculates word familiarity based on a z standardized measure of how many people know a word (from dataset)
    Returns the average word familiarity of all words
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    df = pd.read_excel(dataset_path)
    df['Word'] = df['Word'].str.lower()

    total_words = 0
    total_word_familiarity_values = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            #find word familiarity value for all nouns
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['Word'] == word, 'Pknown']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['Word'] == word_lemma, 'Pknown']

                if not result.empty:
                    total_word_familiarity_values += result.item()
                elif not result_lemma.empty:
                    total_word_familiarity_values += result_lemma.item()
                else:
                    total_words -= 1
    return total_word_familiarity_values / total_words

def age_of_acquisition(nlp=None, file_path=None, dataset_path="datasets/dataset_for_age_of_acquisition.xlsx"):
    """
    Takes in a natural language processor, file path, dataset path
    Calculates age of acquisition using age measure from dataset
    Returns the average age of acquisition for all words
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    df = pd.read_excel(dataset_path)
    df['Word'] = df['Word'].str.lower()

    total_words = 0
    total_AoA_values = 0

    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['Word'] == word, 'AoA']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['Word'] == word_lemma, 'AoA']

                if not result.empty:
                    total_AoA_values += result.item()
                elif not result_lemma.empty:
                    total_AoA_values += result_lemma.item()
                else:
                    total_words -= 1
    return total_AoA_values / total_words

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

def avg_similarity_of_words(nlp=None, file_path=None, window_size=3):
    """
    Takes in a natural language processor and file path, window size(default 3)
    Computes average similarity between words in each window
    Returns average similarity score across all windows
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    word_list = []
    similarity_scores = []
    #make word list
    for token in doc:
        if token.is_alpha:
            word_list.append(token.text)
    #get list of words in window
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

    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    return avg_similarity

def max_similarity_of_words(nlp=None, file_path=None, window_size=3):
    """
    Takes in a natural language processor and file path, window size(default 3)
    Computes average similarity between words in each window
    Returns maximum similarity score across all windows
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    word_list = []
    similarity_scores = []
    #make word list
    for token in doc:
        if token.is_alpha:
            word_list.append(token.text)
    #get list of words in window
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

    max_similarity = max(similarity_scores)
    return max_similarity

def std_similarity_of_words(nlp=None, file_path=None, window_size=3):
    """
    Takes in a natural language processor and file path, window size(default 3)
    Computes standard deviation of similarity between words in each window
    Returns standard deviation similarity score across all windows
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    word_list = []
    similarity_scores = []
    #make word list
    for token in doc:
        if token.is_alpha:
            word_list.append(token.text)
    #get list of words in window
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

    standard_deviation_similarity = statistics.stdev(similarity_scores)
    return standard_deviation_similarity

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

def stats_proportion_coordinators(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Calculates the ratio of coordinators to total words in a sentence
    Returns the average, minimum, maximum, and standard deviation across all sentences
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    coordinators_to_word_ratios = []
    stats_num_coordinators = {}
    for sentence in doc.sents:
        number_of_words = 0
        number_of_coordinators = 0
        for token in sentence:
            if token.is_alpha:
                number_of_words += 1
                if token.pos_ == "CCONJ":
                    number_of_coordinators += 1
        coordinators_to_word_ratios.append(number_of_coordinators / number_of_words)

    stats_num_coordinators["mean"] = sum(coordinators_to_word_ratios) / len(coordinators_to_word_ratios)
    stats_num_coordinators["maximum"] = max(coordinators_to_word_ratios)
    stats_num_coordinators["minimum"] = min(coordinators_to_word_ratios)
    stats_num_coordinators["standard deviation"] = statistics.stdev(coordinators_to_word_ratios)
    return stats_num_coordinators

def stats_proportion_auxiliaries(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Calculates the ratio of auxiliaries to total words in a sentence
    Returns the average, minimum, maximum, and standard deviation across all sentences
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    auxiliaries_to_word_ratios = []
    stats_num_auxiliaries = {}
    for sentence in doc.sents:
        number_of_words = 0
        number_of_auxiliaries = 0
        for token in sentence:
            if token.is_alpha:
                number_of_words += 1
                if token.pos_ == "AUX":
                    number_of_auxiliaries += 1
        auxiliaries_to_word_ratios.append(number_of_auxiliaries / number_of_words)

    stats_num_auxiliaries["mean"] = sum(auxiliaries_to_word_ratios) / len(auxiliaries_to_word_ratios)
    stats_num_auxiliaries["maximum"] = max(auxiliaries_to_word_ratios)
    stats_num_auxiliaries["minimum"] = min(auxiliaries_to_word_ratios)
    stats_num_auxiliaries["standard deviation"] = statistics.stdev(auxiliaries_to_word_ratios)
    return stats_num_auxiliaries

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



def stats_proportion_adjectives(nlp=None, file_path=None):
    """
    Takes in a natural language processor and file path
    Calculates the ratio of adjectives to total words in a sentence
    Returns the average, minimum, maximum, and standard deviation across all sentences
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    adjectives_to_word_ratios = []
    stats_num_adjectives = {}
    for sentence in doc.sents:
        number_of_words = 0
        number_of_adjectives = 0
        for token in sentence:
            if token.is_alpha:
                number_of_words += 1
                if token.pos_ == "ADJ":
                    number_of_adjectives += 1
        adjectives_to_word_ratios.append(number_of_adjectives / number_of_words)

    stats_num_adjectives["mean"] = sum(adjectives_to_word_ratios) / len(adjectives_to_word_ratios)
    stats_num_adjectives["maximum"] = max(adjectives_to_word_ratios)
    stats_num_adjectives["minimum"] = min(adjectives_to_word_ratios)
    stats_num_adjectives["standard deviation"] = statistics.stdev(adjectives_to_word_ratios)
    return stats_num_adjectives



