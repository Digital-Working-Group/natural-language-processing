"""
syntactic_errors.py
Functions related to syntactic errors.
"""
def nonword_frequency(nlp_util, **kwargs):
    """
    Calculates the frequency of non-words per amount of words.
    Non-words are defined as the token's text or lemma not being in the dataset.
    The dataset (datasets/words_alpha.txt) comes from Kaggle and contains over 466K English words.
    """
    dataset_fp = kwargs.get('dataset_fp', 'datasets/words_alpha.txt')
    amount = kwargs.get('amount', 100)
    total_words = 0
    total_nonwords = 0
    word_set = set()
    with open(dataset_fp, "r") as infile:
        for line in infile:
            word_set.add(line.strip())
    for token in nlp_util.doc:
        if token.is_alpha:
            total_words += 1
            if token.text.lower() not in word_set and token.lemma_.lower() not in word_set:
                total_nonwords += 1
    function = 'nonword_frequency'
    parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath, 'dataset_fp': dataset_fp,
        'amount': amount, 'function': function}
    data = {'total_nonwords': total_nonwords, 'total_words': total_words,
        'nonword_frequency': total_nonwords / total_words * amount}
    final_data = {'parameters': parameters, 'data': data}
    nlp_util.write_json_model(function, final_data)
