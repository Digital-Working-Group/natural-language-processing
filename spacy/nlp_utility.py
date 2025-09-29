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
        self.nlp = nlp
        self.doc = self.nlp(self.path_filepath.read_text(encoding='utf-8'))
        self.data = {}

    def write_json_model(self, function, final_data, **kwargs):
        """
        write an output json for a function that uses a spaCy model;
        update self.data if add_to_data is not None.
        """
        quiet = kwargs.get('quiet', False)
        ext = kwargs.get('ext', '')
        add_to_data = kwargs.get('add_to_data', function)
        ext = f'_{ext}' if '_' not in ext and ext != '' else ext
        filename = f'{self.path_filepath.stem}{ext}'
        out_fp = self.path_filepath.parent.joinpath(function, self.model, f'{filename}.json')
        json_save(final_data, out_fp)
        if not quiet:
            print(f'wrote: {out_fp}')
        if add_to_data is not None:
            self.data[function] = final_data

    def write_all_data(self, ext):
        """
        write all self.data to JSON
        """
        out_fp = self.path_filepath.parent.joinpath('data', self.model, ext,
            f'{self.path_filepath.stem}.json')
        json_save(self.data, out_fp)
        print(f'wrote: {out_fp}')
