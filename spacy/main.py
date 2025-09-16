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
    filepath = 'sample_text/test.txt'
    tag_list = ['POS', 'TAG']
    amount = 100
    pos_t.pos_tag_ratio(model, filepath, tag_list, amount=amount)

if __name__ == '__main__':
    main_pos_t()
