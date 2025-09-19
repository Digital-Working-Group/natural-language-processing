"""
main.py
main entrypoint
"""
import pos_tagging as pos_t

def pos_tag_ratio():
    """
    run pos_tagging.pos_tag_ratio()
    """
    model = 'en_core_web_lg'
    tag_list = ['POS', 'TAG']
    amount = 100
    sentence = 'sample_text/sentence.txt'
    paragraph = 'sample_text/paragraph.txt'
    story = 'sample_text/story.txt'
    sample_files = [sentence, paragraph, story]
    for filepath in sample_files:
        pos_t.pos_tag_ratio(model, filepath, tag_list, amount=amount)

def alpha_pos_ratio():
    """
    run pos_tagging.alpha_pos_ratio()
    """
    model = 'en_core_web_lg'
    sentence = 'sample_text/sentence.txt'
    paragraph = 'sample_text/paragraph.txt'
    story = 'sample_text/story.txt'
    sample_files = [sentence, paragraph, story]
    pos_to_list = {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'],
        'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}
    for filepath in sample_files:
        pos_t.alpha_pos_ratio(model, filepath, pos_to_list=pos_to_list)

def alpha_pos_ratio_sentences():
    """
    run pos_tagging.alpha_pos_ratio_sentences()
    """
    model = 'en_core_web_lg'
    sentence = 'sample_text/sentence.txt'
    paragraph = 'sample_text/paragraph.txt'
    story = 'sample_text/story.txt'
    sample_files = [sentence, paragraph, story]
    pos_to_list = {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'],
        'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}
    for filepath in sample_files:
        pos_t.alpha_pos_ratio_sentences(model, filepath, pos_to_list=pos_to_list)

def main():
    """
    main entrypoint
    """
    pos_tag_ratio()
    alpha_pos_ratio()
    alpha_pos_ratio_sentences()

if __name__ == '__main__':
    main()
