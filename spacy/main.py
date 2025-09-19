"""
main.py
main entrypoint
"""
import pos_tagging as pos_t

def main_pos_t():
    """
    run scripts from pos_tagging.py
    """
    model = 'en_core_web_lg'
    tag_list = ['POS', 'TAG']
    amount = 100
    sentence = 'sample_text/sentence.txt'
    paragraph = 'sample_text/paragraph.txt'
    story = 'sample_text/story.txt'
    for filepath in [sentence, paragraph, story]:
        pos_t.pos_tag_ratio(model, filepath, tag_list, amount=amount)

if __name__ == '__main__':
    main_pos_t()
