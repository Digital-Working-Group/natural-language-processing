"""
nlp_utility.py
Utility class to hold various attributes
"""
from pathlib import Path
import spacy
from utility import json_save

class NLPUtil:
    """
    class to hold various attributes for utility and writing output files
    """
    def __init__(self, model, filepath, **kwargs):
        """
        constructor
        """
        pipe_list = kwargs.get('pipe_list', [])
        self.model = model
        self.filepath = filepath
        self.path_filepath = Path(filepath)
        nlp = spacy.load(model)
        for pipe in pipe_list:
            nlp.add_pipe(pipe)
        self.doc = nlp(self.path_filepath.read_text(encoding='utf-8'))

    def write_json_model(self, function, final_data, **kwargs):
        """
        write an output json for a function that uses a spaCy model
        """
        quiet = kwargs.get('quiet', False)
        ext = kwargs.get('ext', '')
        ext = f'_{ext}' if '_' not in ext and ext != '' else ext
        filename = f'{self.path_filepath.stem}{ext}'
        out_fp = self.path_filepath.parent.joinpath(function, self.model, f'{filename}.json')
        json_save(final_data, out_fp)
        if not quiet:
            print(f'wrote: {out_fp}')
