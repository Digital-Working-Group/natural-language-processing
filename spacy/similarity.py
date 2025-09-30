"""
similarity.py
Functions related to text similarity.
"""
import itertools
import statistics

def get_similarity(nlp, first, second):
    """
    get similarity of two tokens
    """
    first = nlp(str(first))
    second = nlp(str(second))
    return round(first.similarity(second), 2)

def get_similarity_stats(similarity_scores):
    """
    get mean, max, min, std
    """
    return {
        "avg_similarity_score": sum(similarity_scores) / len(similarity_scores),
        "max_similarity_score": max(similarity_scores),
        "min_similarity_score": min(similarity_scores),
        "std_similarity_score": statistics.stdev(similarity_scores),
        }

def doc_similarity(nlp_util, window_size=3):
    """
    calculates word similarity across a sliding window over a document's alphanumeric tokens.
    """
    word_list = [t.text.lower() for t in nlp_util.doc if t.is_alpha]
    similarity_scores = []
    windows = []
    for idx in range(len(word_list) - window_size + 1):
        window_similarity_scores = []
        words_in_window = word_list[idx: idx + window_size]
        for first, second in itertools.combinations(words_in_window, 2):
            window_similarity_scores.append(get_similarity(nlp_util.nlp, first, second))
        windows.append(get_similarity_stats(window_similarity_scores))
        similarity_scores.append(windows[idx]['avg_similarity_score'])
    data = get_similarity_stats(similarity_scores)
    data.update({'num_windows': len(similarity_scores), 'windows': windows})
    function = 'doc_similarity'
    parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath,
        'window_size': window_size, 'function': function}
    final_data = {'parameters': parameters, 'data': data}
    nlp_util.write_json_model(function, final_data)

def sent_similarity(nlp_util):
    """
    calculates word similarity of sentences over a document's alphanumeric tokens.
    """
    sentence_list = list(nlp_util.doc.sents)
    similarity_scores = []
    for first, second in itertools.combinations(sentence_list, 2):
        first = " ".join([t.text.lower() for t in first if t.is_alpha])
        second = " ".join([t.text.lower() for t in second if t.is_alpha])
        similarity_scores.append(get_similarity(nlp_util.nlp, first, second))
    data = get_similarity_stats(similarity_scores)
    data.update({'num_sentences': len(sentence_list)})
    function = 'sent_similarity'
    parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath,
        'function': function}
    final_data = {'parameters': parameters, 'data': data}
    nlp_util.write_json_model(function, final_data)
