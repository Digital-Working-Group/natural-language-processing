"""
main.py
main entrypoint
"""
from pathlib import Path
import spacy
import pos_tagging as pos_t

def main_pos_t():
    """
    run scripts from pos_tagging.py
    """
    nlp = spacy.load('en_core_web_lg')
    file_path = 'sample_text/test.txt'
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    doc_df = pos_t.data_to_df(doc)
    tag_counts = pos_t.pos_tag_counts(doc_df, 'POS', amount=100)
    print(tag_counts)

if __name__ == '__main__':
    main_pos_t()
