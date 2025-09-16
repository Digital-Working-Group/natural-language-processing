"""
utility.py
utility functions, such as writing JSON output
"""
import os
import json

def json_save(data, out_fp, sort_keys=False, indent=4):
    """
    save data to a JSON file
    """
    parent = os.path.dirname(out_fp)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    with open(out_fp, 'w') as out_file:
        json.dump(data, out_file, sort_keys=sort_keys, indent=indent)
