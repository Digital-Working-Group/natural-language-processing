import spacy
import pathlib
import pandas as pd
from lexical_diversity import lex_div as ld
from spellchecker import SpellChecker
import itertools

nlp = spacy.load('en_core_web_lg')

def data_to_df(doc):
    '''Takes in a spacy doc object and returns a pandas dataframe with token attributes '''
    d = []
    for token in doc:
        d.append({
            "TEXT": token.text,
            "LEMMA": token.lemma_,
            "pip instaPOS": token.pos_,
            "TAG": token.tag_,
            "DEP": token.dep_,
            "SHAPE": token.shape_,
            "ALPHA": token.is_alpha,
            "STOP": token.is_stop
        })
    return pd.DataFrame(d)


def tag_ratio(doc, tag='POS', amount=100):
    ''' Takes in spacy doc object, desired tag, and per word amount(default is 100)
        Returns a dictionary with pos tags and their proportions'''
    df = data_to_df(doc)
    tag_count_per_amount = {}
    total = 0
    #counts tags
    for tag in df[tag]:
        if tag not in tag_count_per_amount:
            tag_count_per_amount[tag] = 1
        else:
            tag_count_per_amount[tag] += 1
        total += 1
    #calculate proportion
    for tag in tag_count_per_amount:
        tag_count_per_amount[tag] = tag_count_per_amount[tag] / total * amount

    return tag_count_per_amount

def proportion_tense_inflected_verbs(doc, amount=100):
    '''
    Takes in a spacy doc object and per word amount(default is 100)
    Calculates proportion of tense inflected verbs based on number of tense inflected verbs
    Returns the proportion of tense inflected verbs
    '''
    total_words = 0
    present_verbs = []
    past_verbs = []
    modal_auxiliary = []
    num_tiv = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            #find tense inflected verbs
            if token.pos_ == "VERB" and token.morph.get("Tense") == ["Pres"]:
                present_verbs.append(token.morph)
            if token.pos_ == "VERB" and token.morph.get("Tense") == ["Past"]:
                past_verbs.append(token.morph)
            if token.pos_ == "AUX" and token.tag_ == "MD":
                modal_auxiliary.append(token.morph)

            num_tiv = len(present_verbs) + len(modal_auxiliary) + len(past_verbs)
    return num_tiv / total_words * amount

def calculate_idea_density(doc, amount=1):
    '''
    Takes in a spacy doc object and per word amount(default is 1))
    Calculates idea density - the number of proportions divided by the total words
    Returns the idea density
    '''
    count_props = 0
    idea_density_sentences = 0
    num_sentences = 0
    idea_density = 0
    for sent in doc.sents:
        num_sentences += 1
        count_words = 0
        for token in sent:
            if token.is_alpha:
                count_words += 1
                if token.pos_ == "PROPN":
                    count_props += 1
        idea_density_sentences += count_props / count_words
    idea_density += idea_density_sentences / num_sentences * amount
    return idea_density

def abstractness(doc, amount=100):
    """
    Takes in a spacy doc object and per word amount(default is 100)
    Calculates abstractness - inverse of concreteness value taken from dataset
    Returns the average abstractness per word amount
    """
    df = pd.read_excel('13428_2013_403_MOESM1_ESM.xlsx')
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
    return total_abstractness_values / total_words * amount

def semantic_ambiguity(doc, amount=100):
    """
    Takes in a spacy doc object and per word amount(default is 100)
    Calculates semantic ambiguity using semantic diversity value from dataset
    Returns the semantic ambiguity per word amount
    """
    df = pd.read_excel("Semantic_diversity.xlsx", skiprows=1)
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
    return total_ambiguity_values / total_words * amount

def word_frequency(doc, amount=100):
    """
    Takes in a spacy doc object and per word amount(default is 100)
    Calculates word frequency using log10 word frequency measure from dataset
    Returns the word frequency per word amount
    """
    df = pd.read_excel("SUBTLEXusExcel2007.xlsx")
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
    return total_wf_values / total_words * amount

def word_prevalence(doc, amount=100):
    """
    Takes in a spacy doc object and per word amount(default is 100)
    Calculates word prevalence using prevalence measure from dataset
    Returns the word prevalence per word amount
    """
    df = pd.read_excel("word_prevelance.xlsx")
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
    return total_wp_values / total_words * amount


def word_familiarity(doc, amount=100):
    """
    Takes in a spacy doc object and per word amount(default is 100)
    Calculates word familiarity based on a z standardized measure of how many people know a word (from dataset)
    Returns the word familiarity per word amount
    """
    df = pd.read_excel("word_prevelance.xlsx")
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
    return total_word_familiarity_values / total_words * amount

def age_of_acquisition(doc, amount=100):
    """
    Takes in a spacy doc object and per word amount(default is 100)
    Calculates age of acquisition using age measure from dataset
    Returns the age of acquisition per word amount
    """
    df = pd.read_excel("for_AoA.xlsx")
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
    return total_AoA_values / total_words * amount

def frequency_nonwords(doc, amount=100):
    """
    Takes in a spacy doc object and per word amount(default is 100)
    Calculates frequency of non-words using dataset of english words
    """
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


def length_of_sentences(doc, amount=1):
    """
    Takes in a spacy doc object and per word amount(default is 1)
    Calculates average length of sentences
    Returns the average length of sentences per word amount
    """
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
    return avg_words_per_sentence * amount

#Occurances of the most frequent token in the text
def occurrences_of_most_frequent(doc):
    """
    Takes in a spacy doc object
    Finds most frequent word and counts its occurrences
    Returns most frequent word and occurrences
    """
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

def mattr_automatic(doc):
    """
    Takes in a spacy doc object
    uses ld.mattr from the python library "Lexical diversity" to automatically calculate moving average type token ratio
    Returns moving average type token ratio
    """
    word_list = []
    for token in doc:
        if token.is_alpha:
            word_list.append(token.text.lower())

    return ld.mattr(word_list, window_length=25)

def moving_average_text_token_ratio(doc, window_size=20):
    """
    Takes in a spacy doc object and window size (default is 20)
    Calculates type/token ratio for a fixed-length window, and averages type/token ratios from all windows.
    Returns moving average text token ratio
    """
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

def term_frequency(doc, term):
    """
    Takes in a spacy doc object and target string
    Returns frequency of target string in document
    """
    term_appearances = 0
    total_words = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
        if token.text.lower() == term:
            term_appearances += 1
    return term_appearances / total_words

def inverse_document_frequency(document_list, term):
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

def tf_idf(document_list, doc, term):
    """
    Takes in a document list, spacy doc object, and target string
    Calculates TF-IDF by multiplying TF by IDF
    returns TF-IDF value
    """
    return term_frequency(doc, term) * inverse_document_frequency(document_list, term)

def repeating_unique_word_ratio(doc, amount=100):
    """
    Takes in a spacy doc object and per word amount
    calculates (number of repeating words) / (number of unique words)
    Returns ratio of repeating to unique words per word amount
    """
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

def incorrectly_followed_articles(doc):
    """
    Takes in a spacy doc object
    counts number of articles not followed by a noun, proper noun, or adjective
    returns the count
    """
    articles = ["a", "an", "the"]
    count = 0
    for i, token in enumerate(doc):
        if token.text.lower() in articles:
            following_token = doc[i+1]
            if following_token.pos_ not in ["ADJ", "NOUN", "PROPN"]:
                count += 1
    return count

def number_of_unique_tokens(doc):
    """
    Takes in a spacy doc and returns the number of unique tokens
    """
    unique_tokens = set()
    for token in doc:
        unique_tokens.add(token.text)
    return len(unique_tokens)

def number_of_unique_lemmas(doc):
    """
    Takes in a spacy doc and returns the number of unique lemmas
    """
    lemmas = set()
    for token in doc:
        lemmas.add(token.lemma_)
    return len(lemmas)

def ratio_of_nouns(doc):
    """
    Takes in a spacy doc and returns the ratio of nouns to total words
    """
    total_words = 0
    total_nouns = 0
    for token in doc:
        if token.is_alpha:
            print(token)
            total_words += 1
            if token.pos_== "NOUN":
                total_nouns += 1
    return total_nouns / total_words

spell = SpellChecker()
def count_nonwords_with_spellcheck(doc):
    """
    Takes in a spacy doc and returns the number of nonwords
    """
    number_of_nonwords = 0
    nonwords = []
    for token in doc:
        if token.is_alpha and token.text.lower() not in spell:
            number_of_nonwords += 1
            nonwords.append(token.text)
    return number_of_nonwords, nonwords

def avg_wh_words(doc, amount=100):
    """
    Takes in a spacy doc and per word amount
    calculates number of wh words per word amount
    returns average number of wh-words per word amount
    """
    total_wh_words = 0
    total_words = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            word = token.text.lower()
            if word[0:2] == "wh":
                total_wh_words += 1
    return total_wh_words / total_words * amount

def avg_num_nonwords(doc, amount=100):
    """
    Takes in doc object and word amount
    Counts number of nonwords by checking if words are in spacys vocabulary
    Returns average number of nonwords per word amount
    """
    nonwords = 0
    total_words = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.is_oov:
                nonwords += 1
    return nonwords / total_words * amount

def mean_similarity_of_sentences(doc, amount=100):
    """
    Takes in a spacy doc object and per word amount
    calculates mean similarity of all combinations of sentences
    Returns mean similarity per word amount
    """
    sentence_list = list(doc.sents)
    similarity_scores = []
    for comb in itertools.combinations(sentence_list, 2):
        sentence_one = str(comb[0])
        sentence_two = str(comb[1])
        doc_one = nlp(sentence_one)
        doc_two = nlp(sentence_two)
        similarity_score = round(doc_one.similarity(doc_two), 2)
        similarity_scores.append(similarity_score)
    return sum(similarity_scores) / len(similarity_scores) * amount

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

def avg_dependency_tree_height(doc):
    """
    Takes in a spacy doc object
    Uses tree_height() to calculate max height of dependency trees
    Returns average height of all dependency trees
    """
    tree_depths = []
    max_depth = 0
    for sentence in doc.sents:
        for token in sentence:
            if token.dep_ == "ROOT":
                max_depth = tree_height(token)
        tree_depths.append(max_depth)
    return sum(tree_depths) / len(tree_depths)

def max_dependency_tree_height(doc):
    """
    Takes in a spacy doc object
    Uses tree_height() to calculate max height of dependency trees
    Returns average height of all dependency trees
    """
    tree_depths = []
    max_depth = 0
    for sentence in doc.sents:
        for token in sentence:
            if token.dep_ == "ROOT":
                max_depth = tree_height(token)
        tree_depths.append(max_depth)
    return max(tree_depths)
