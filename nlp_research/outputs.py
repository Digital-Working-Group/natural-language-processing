import spacy
import pathlib
import pandas as pd
from nlp_research.nlp_functions import data_to_df, num_tense_inflected_verbs, idea_density, abstractness, \
    semantic_ambiguity, word_frequency, word_prevelance, word_familiarity, age_of_aquisition, semantic_ambiguity, \
    frequency_nonwords
from nlp_functions import POS_ratio

nlp = spacy.load('en_core_web_lg')
file_name = "test.txt"
doc = nlp(pathlib.Path("test.txt").read_text(encoding="utf-8"))

#Dataframe
print(data_to_df(doc).head())
print("\n\n")
#POS Ratio
print(f"pos: {POS_ratio(doc, 'TAG',100)}")
print(f"lemma: {POS_ratio(doc, 'LEMMA',100)}")
print(f"text: {POS_ratio(doc, 'TEXT',100)}")
print(f"tag: {POS_ratio(doc, 'TAG',100)}")
print(f"dep: {POS_ratio(doc, 'DEP',100)}")
print(f"shape: {POS_ratio(doc, 'SHAPE',100)}")
print(f"is alpha? : {POS_ratio(doc, 'ALPHA',100)}")
print(f"is stop? : {POS_ratio(doc, 'STOP',100)}")

#Tense Inflected Verbs
print(f"number of tense inflected verbs: {num_tense_inflected_verbs(doc)}")

#Idea Density
print(f"Idea Density is: {idea_density(doc)}")

#Abstractness
print(f"abstract: {abstractness(doc)}")

#Semantic Ambiguity
print(f"semantic ambiguity: {semantic_ambiguity(doc)}")

#Word frequency lg10
print(f"word frequency: {word_frequency(doc)}")

#Word prevelance
print(f"word prevelance: {word_prevelance(doc)}")

#Word Familiarity
print(f"word familiarity: {word_familiarity(doc)}")

#Age of Aquisition
print(f"age of aquisition: {age_of_aquisition(doc)}")

#frequency of non-words
doc2 = nlp(pathlib.Path("contains_nonwords.txt").read_text(encoding="utf-8"))
print(f"frequency of nonwords: {frequency_nonwords(doc2, 1)}")
