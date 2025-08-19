from pos_tagging import data_to_df, tag_ratio, pos_tag_counts
from semantic_complexity import semantic_ambiguity
from syntactic_complexity import sentence_lengths
from syntactic_complexity import num_tense_inflected_verbs
from semantic_complexity import calculate_idea_density
from semantic_complexity import abstractness, word_frequency, word_prevalence, word_familiarity, age_of_acquisition
from syntactic_errors import nonword_frequency
from lexical_repetition import most_frequent_word
import spacy

def main():
    """
    Main entry point for extracting linguistic features.
    """
    #tokens and attributes visualized in a dataframe
    print(data_to_df(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt").head())

    #Universal POS tags and on average how many of each are present per 100 words
    print(pos_tag_counts(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", tag="POS", amount=100))

    #Detailed POS tags and on average how many of each are present per 100 words
    print(pos_tag_counts(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", tag="TAG", amount=100))

    #parts-of-speech tagging
    print(tag_ratio(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", amount=100))

    #number of tense inflected verbs per 100 words
    print(num_tense_inflected_verbs(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", amount=100))

    #sentences and their idea density
    print(calculate_idea_density(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #average abstractness value across all nouns
    print(abstractness(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #average semantic ambiguity value across all nouns
    print(semantic_ambiguity(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #average word frequency value across all nouns
    print(word_frequency(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #average word prevalence value across all nouns
    print(word_prevalence(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #average word familiarity value across all nouns
    print(word_familiarity(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #average age of acquisition value across all nouns
    print(age_of_acquisition(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #average number of non-words per 100 words
    print(nonword_frequency(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/contains_nonwords.txt", dataset_fp="words_alpha.txt", amount=100))

    #list of sentence lengths in document
    print(sentence_lengths(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #most frequent word and its number of occurrences
    print(most_frequent_word(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

if __name__ == '__main__':
    main()