import spacy
import claucy
import math
import pathlib
from spellchecker import SpellChecker
import itertools
from spacy import displacy
import pandas as pd
from collections import defaultdict

nlp = spacy.load('en_core_web_lg')
doc = nlp("I am going to cut you Jack. That is why yes.")


def proportion_tense_inflected_verbs(doc, amount=100):
    '''
    Takes in a spacy doc object and per word amount(default is 100)
    Filters on tokens that only contain alphabetic characters (excluding punctuation or numbers), and calculates proportion of tense inflected verbs based on number of tense inflected verbs
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
            print(token.text, token.pos_, token.morph.get("Tense"))
            if token.pos_ == "VERB" and "Pres" in token.morph.get("Tense"):
                present_verbs.append(token.morph)
            if token.pos_ == "VERB" and "Past" in token.morph.get("Tense"):
                past_verbs.append(token.morph)
            if token.pos_ == "AUX" and token.tag_ == "MD":
                modal_auxiliary.append(token.morph)

            num_tiv = len(present_verbs) + len(modal_auxiliary) + len(past_verbs)
    return num_tiv / total_words * amount

print(proportion_tense_inflected_verbs(doc))

print(spacy.explain("IN"))















