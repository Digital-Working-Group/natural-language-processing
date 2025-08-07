import spacy
import pathlib
import pandas as pd

from nlp_research.count_pos import max_similarity_of_words
from nlp_research.nlp_functions import data_to_df, num_tense_inflected_verbs, calculate_idea_density, \
    abstractness, \
    semantic_ambiguity, word_frequency, word_prevalence, word_familiarity, age_of_acquisition, semantic_ambiguity, \
    frequency_nonwords, length_of_sentences, occurrences_of_most_frequent, \
    moving_average_text_token_ratio, \
    incorrectly_followed_articles, number_of_unique_tokens, avg_wh_words, avg_num_nonwords, \
    count_nonwords_with_spellcheck, mean_similarity_of_sentences, avg_dependency_tree_height, \
    max_dependency_tree_height, avg_similarity_of_words, \
    max_similarity_of_words, std_similarity_of_words, ratio_of_pronouns, ratio_of_conjunctions, \
    stats_proportion_coordinators, stats_proportion_auxiliaries, stats_proportion_subjects, \
    count_num_sentences_without_verbs, total_consecutive_words, stats_proportion_adjectives
from nlp_functions import tag_ratio

nlp = spacy.load('en_core_web_lg')
file_name = "test.txt"
doc = nlp(pathlib.Path("test.txt").read_text(encoding="utf-8"))
doc1 = nlp("contains_nonwords.txt")

#Dataframe
print(data_to_df(doc).head())
print("\n\n")
#POS Ratio
print(f"pos: {tag_ratio(tag='TAG',amount=100)}")
print(f"lemma: {tag_ratio(tag='LEMMA',amount=100)}")
print(f"text: {tag_ratio(tag='TEXT',amount=100)}")
print(f"tag: {tag_ratio(tag='TAG',amount=100)}")
print(f"dep: {tag_ratio(tag='DEP',amount=100)}")
print(f"shape: {tag_ratio(tag='SHAPE',amount=100)}")
print(f"is alpha? : {tag_ratio(tag='ALPHA',amount=100)}")
print(f"is stop? : {tag_ratio(tag='STOP',amount=100)}")

#Tense Inflected Verbs
print(f"number of tense inflected verbs: {num_tense_inflected_verbs()}")

#Idea Density
print(f"Idea Density is: {calculate_idea_density()}")

#Abstractness
print(f"abstract: {abstractness()}")

#Semantic Ambiguity
print(f"semantic ambiguity: {semantic_ambiguity()}")

#Word frequency lg10
print(f"word frequency: {word_frequency()}")

#Word prevalence
print(f"word prevalence: {word_prevalence()}")

#Word Familiarity
print(f"word familiarity: {word_familiarity()}")

#Age of Acquisition
print(f"age of acquisition: {age_of_acquisition()}")

#frequency of non-words
print(f"frequency of nonwords: {frequency_nonwords(file_path="contains_nonwords.txt")}")

#number of words in sentence
print(f"avg number of words in a sentence: {length_of_sentences()}")

#occurences of most frequent
print(f"occurrences of most frequent word: {occurrences_of_most_frequent()}")

#lexical diversity
print(f"manual lexical diversity: {moving_average_text_token_ratio(doc)}")

#number of incorrectly followed articles
print(f"number of incorrectly followed articles: {incorrectly_followed_articles(doc)}")

#number of unique tokens
print(f"number of unique tokens: {number_of_unique_tokens(doc)}")

#number of unique lemmas
print(f"number of unique tokens: {number_of_unique_tokens(doc)}")

#number of nonwords
print(f"number of nonwords: {count_nonwords_with_spellcheck(file_path="contains_nonwords.txt")}")

#Average number of wh words
print(f"Average number of wh words: {avg_wh_words()})")

#Average number of nonwords
print(f"Average number of non-words: {avg_num_nonwords(file_path="contains_nonwords.txt")}")

#Average similarity of sentences
print(f"Average semantic similarity: {mean_similarity_of_sentences()}")

#Average height of dependency trees
print(f"Average height of dependency trees: {avg_dependency_tree_height()}")

#Max height of dependency trees
print(f"Max height of dependency trees: {max_dependency_tree_height()}")

#Average similarity of words across moving windows
print(f"average similarity of words with moving window size of 3: {avg_similarity_of_words()} ")

#Maximum similarity of words across moving windows
print(f"Maximum similarity of words with moving window size of 3: {max_similarity_of_words()})")

#Standard deviation of similarity of words acros moving windows
print(f"Standard deviation of similarity of words with moving window size of 3: {std_similarity_of_words()})")

#Ratio of pronouns to total words
print(f"ratio of pronouns to total words: {ratio_of_pronouns()}")

#Ratio of conjunctions to total words
print(f"Ratio of conjunctions to total words: {ratio_of_conjunctions()}")

#Minimum, maximum, and standard deviation of proportion of coordinators across sentences
print(f"Minimum, maximum, and standard deviation of proportion of coordinators across sentences: {stats_proportion_coordinators()}")

#Minimum, maximum, and standard deviation of proportion of auxiliaries across sentences
print(f"Minimum, maximum, and standard deviation of proportion of auxiliaries across sentences: {stats_proportion_auxiliaries()}")

#Minimum, maximum, and standard deviation of proportion of subjects across sentences
print(f"Minimum, maximum, and standard deviation of proportion of subjects across sentences: {stats_proportion_subjects()}")

#Count the number of sentences without verbs
print(f"number of sentences without verbs: {count_num_sentences_without_verbs()}")

#count the number of consecutively repeating words
print(f"Number of consecutive repeating words: {total_consecutive_words()}")

#Minimum, maximum, and standard deviation of proportion of adjectives across sentences
print(f"Minimum, maximum, and standard deviation of proportion of adjectives across sentences: {stats_proportion_adjectives()}")