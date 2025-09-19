"""
utility.py
utility functions, such as writing JSON output
"""
import os
import json
from pathlib import Path

def write_json_model(path_filepath, function, model, final_data):
    """
    write an output json for a function that uses a spaCy model
    """
    out_fp = path_filepath.parent.joinpath(function, model,
        Path(path_filepath.name).with_suffix('.json'))
    json_save(final_data, out_fp)

def json_save(data, out_fp, sort_keys=False, indent=4):
    """
    save data to a JSON file
    """
    parent = os.path.dirname(out_fp)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    with open(out_fp, 'w') as out_file:
        json.dump(data, out_file, sort_keys=sort_keys, indent=indent)
