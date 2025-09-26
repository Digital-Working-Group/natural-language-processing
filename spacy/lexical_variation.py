"""
lexical_variation.py
Functions related to lexical variation
"""
import utility as util

def get_window_data(word_list, window_size):
    """
    get data from each window
    """
    num_word_list = len(word_list)
    assert num_word_list >= window_size, 'The number of words in the word_list '+\
        f'({num_word_list}) is less than the window_size ({window_size})'
    total_type_token_ratio = 0
    windows = []
    for idx in range(num_word_list - window_size + 1):
        words_in_window = word_list[idx: idx + window_size]
        unique_words = set(words_in_window)
        num_words = len(words_in_window)
        num_unique_words = len(unique_words)
        type_token_ratio = num_unique_words / num_words
        total_type_token_ratio += type_token_ratio
        windows.append({'num_unique_words': num_unique_words,
            'type_token_ratio': type_token_ratio})
    return windows, total_type_token_ratio

def moving_type_token_ratio(model, filepath, window_size=20):
    """
    Iterates over alphanumeric tokens only.
    Calculates type/token ratio for a fixed-length window, moving one word at a time.
    Calculates the average of the type/token ratios from all windows.
    
    Raises an AssertionError if the total number of alphanumeric tokens are less than
    the window_size.

    Returns data per window and overall statistics
    data:
        average_type_token_ratio
        num_windows
        window_data: [
            {'num_unique_words', 'type_token_ratio'}
        ]
    """
    doc, path_filepath = util.get_doc_and_filepath(model, filepath)
    word_list = [t.text.lower() for t in doc if t.is_alpha]

    windows, total_type_token_ratio = get_window_data(word_list, window_size)
    num_windows = len(windows)
    average_type_token_ratio = total_type_token_ratio / num_windows
    data = {'average_type_token_ratio': average_type_token_ratio, 'num_windows': num_windows,
        'windows': windows}
    function = 'moving_type_token_ratio'
    parameters = {'model': model, 'filepath': filepath, 'window_size': window_size,
        'function': function}
    final_data = {'parameters': parameters, 'data': data}
    util.write_json_model(path_filepath, function, model, final_data)
