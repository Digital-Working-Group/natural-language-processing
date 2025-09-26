"""
main.py
main entrypoint
"""
import textdescriptives
import pos_tagging as pos_t
import semantic_complexity as sem_c
import syntactic_complexity as syn_c
import lexical_variation as lex_v
from nlp_utility import NLPUtil

def get_sample_files():
    """
    get list of sample files
    """
    sentence = 'sample_text/sentence.txt'
    paragraph = 'sample_text/paragraph.txt'
    story = 'sample_text/story.txt'
    return [sentence, paragraph, story]

def main():
    """
    run all the functions
    """
    model = 'en_core_web_lg'
    sample_files = get_sample_files()
    for filepath in sample_files:
        nlp_util = NLPUtil(model, filepath)

        tag_list = ['POS', 'TAG']
        pos_t.pos_tag_ratio(nlp_util, tag_list, amount=100)

        pos_to_list = {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'],
            'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}
        pos_t.alpha_pos_ratio(nlp_util, pos_to_list=pos_to_list)

        pos_to_list = {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'],
            'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}
        pos_t.alpha_pos_ratio(nlp_util, pos_to_list=pos_to_list)

        sem_c.idea_density_sentences(nlp_util)

        sem_c.abstractness(nlp_util)
        sem_c.age_of_acquisition(nlp_util)
        sem_c.semantic_ambiguity(nlp_util)
        sem_c.word_familiarity(nlp_util)
        sem_c.word_frequency(nlp_util)
        sem_c.word_prevalence(nlp_util)

# def tense_inflected_verbs():
#     """
#     run syntactic_complexity.tense_inflected_verbs()
#     """
#     model = 'en_core_web_lg'
#     sample_files = get_sample_files()
#     for filepath in sample_files:
#         syn_c.tense_inflected_verbs(nlp_util)

# def dependency_distance():
#     """
#     run syntactic_complexity.dependency_distance()
#     """
#     model = 'en_core_web_lg'
#     sample_files = get_sample_files()
#     for filepath in sample_files:
#         syn_c.dependency_distance(nlp_util)

# def moving_type_token_ratio():
#     """
#     run lexical_variation.moving_type_token_ratio()
#     """
#     model = 'en_core_web_lg'
#     sample_files = get_sample_files()
#     for filepath in sample_files:
#         lex_v.moving_type_token_ratio(nlp_util)

if __name__ == '__main__':
    main()
