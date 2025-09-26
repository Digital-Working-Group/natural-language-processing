"""
utility.py
utility functions, such as writing JSON output
"""
import os
import json
from pathlib import Path
import spacy

# def get_doc_and_filepath_add_pipes(model, filepath, pipe_list):
#     """
#     add pipes, then return doc and filepath
#     pipe_list: ['textdescriptives/dependency_distance']
#     """
#     path_filepath = Path(filepath)
#     nlp = spacy.load(model)
#     for pipe in pipe_list:
#         nlp.add_pipe(pipe)
#     doc = nlp(path_filepath.read_text(encoding='utf-8'))
#     return doc, path_filepath

# def get_doc_and_filepath(model, filepath):
#     """
#     get spacy doc and Path(filepath)
#     """
#     path_filepath = Path(filepath)
#     doc = spacy.load(model)(path_filepath.read_text(encoding='utf-8'))
#     return doc, path_filepath

def json_save(data, out_fp, sort_keys=False, indent=4):
    """
    save data to a JSON file
    """
    parent = os.path.dirname(out_fp)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    with open(out_fp, 'w') as out_file:
        json.dump(data, out_file, sort_keys=sort_keys, indent=indent)
