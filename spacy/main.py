"""
main.py
main entrypoint
"""
import textdescriptives
import pos_tagging as pos_t
import semantic_complexity as sem_c
import syntactic_complexity as syn_c
import lexical_variation as lex_v

def get_sample_files():
    """
    get list of sample files
    """
    sentence = 'sample_text/sentence.txt'
    paragraph = 'sample_text/paragraph.txt'
    story = 'sample_text/story.txt'
    return [sentence, paragraph, story]

def pos_tag_ratio():
    """
    run pos_tagging.pos_tag_ratio()
    """
    model = 'en_core_web_lg'
    tag_list = ['POS', 'TAG']
    amount = 100
    sample_files = get_sample_files()
    for filepath in sample_files:
        pos_t.pos_tag_ratio(model, filepath, tag_list, amount=amount)

def alpha_pos_ratio():
    """
    run pos_tagging.alpha_pos_ratio()
    """
    model = 'en_core_web_lg'
    sample_files = get_sample_files()
    pos_to_list = {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'],
        'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}
    for filepath in sample_files:
        pos_t.alpha_pos_ratio(model, filepath, pos_to_list=pos_to_list)

def alpha_pos_ratio_sentences():
    """
    run pos_tagging.alpha_pos_ratio_sentences()
    """
    model = 'en_core_web_lg'
    sample_files = get_sample_files()
    pos_to_list = {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'],
        'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}
    for filepath in sample_files:
        pos_t.alpha_pos_ratio_sentences(model, filepath, pos_to_list=pos_to_list)

def idea_density_sentences():
    """
    run semantic_complexity.idea_density_sentences()
    """
    model = 'en_core_web_lg'
    sample_files = get_sample_files()
    for filepath in sample_files:
        sem_c.idea_density_sentences(model, filepath)

def generate_noun_features():
    """
    run the several generate_noun_feature-based functions from semantic_complexity.py
    """
    model = 'en_core_web_lg'
    sample_files = get_sample_files()
    for filepath in sample_files:
        sem_c.abstractness(model, filepath)
        sem_c.semantic_ambiguity(model, filepath)
        sem_c.word_frequency(model, filepath)
        sem_c.word_prevalence(model, filepath)
        sem_c.word_familiarity(model, filepath)
        sem_c.age_of_acquisition(model, filepath)

def tense_inflected_verbs():
    """
    run syntactic_complexity.tense_inflected_verbs()
    """
    model = 'en_core_web_lg'
    sample_files = get_sample_files()
    for filepath in sample_files:
        syn_c.tense_inflected_verbs(model, filepath)

def dependency_distance():
    """
    run syntactic_complexity.dependency_distance()
    """
    model = 'en_core_web_lg'
    sample_files = get_sample_files()
    for filepath in sample_files:
        syn_c.dependency_distance(model, filepath)

def moving_type_token_ratio():
    """
    run lexical_variation.moving_type_token_ratio()
    """
    model = 'en_core_web_lg'
    sample_files = get_sample_files()
    for filepath in sample_files:
        lex_v.moving_type_token_ratio(model, filepath)

def main():
    """
    main entrypoint
    """
    # pos_tag_ratio()
    # alpha_pos_ratio()
    # alpha_pos_ratio_sentences()
    # idea_density_sentences()
    # generate_noun_features()
    # tense_inflected_verbs()
    # dependency_distance()
    moving_type_token_ratio()

if __name__ == '__main__':
    main()
