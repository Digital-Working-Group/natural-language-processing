from pos_tagging import data_to_df, tag_ratio, pos_tag_counts, ratio_of_nouns, ratio_of_pronouns, ratio_of_conjunctions
from semantic_complexity import semantic_ambiguity
from lexical_repetition import repeating_unique_word_ratio
from lexical_repetition import total_consecutive_words
from syntactic_errors import incorrectly_followed_articles
from syntactic_complexity import sentence_lengths
from syntactic_complexity import num_tense_inflected_verbs, dependency_tree_heights
from semantic_complexity import calculate_idea_density
from semantic_complexity import abstractness, word_frequency, word_prevalence, word_familiarity, age_of_acquisition
from syntactic_errors import nonword_frequency, count_num_sentences_without_verbs
from lexical_repetition import most_frequent_word
from lexical_variation import windowed_text_token_ratio, number_of_unique_lemmas, number_of_unique_tokens
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

    #moving average text token ratio
    print(windowed_text_token_ratio(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", window_size=20))

    #Ratio of repeating words to unique words in the text
    print(repeating_unique_word_ratio(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #number of incorrectly followed articles present in the text
    print(incorrectly_followed_articles(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #list of heights of all dependency trees
    print(dependency_tree_heights(nlp=(spacy.load('en_core_web_lg')), file_path="sample_text/test.txt"))

    #ratio of nouns to total words
    print(ratio_of_nouns(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    # ratio of pronouns to total words
    print(ratio_of_pronouns(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    # ratio of conjunctions to total words
    print(ratio_of_conjunctions(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #number of unique tokens in the text
    print(number_of_unique_tokens(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #number of unique lemmas in the text
    print(number_of_unique_lemmas(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #number of sentences in the text not containing verbs
    print(count_num_sentences_without_verbs(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

    #number of consecutive repeating words
    print(total_consecutive_words(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))
if __name__ == '__main__':
    main()