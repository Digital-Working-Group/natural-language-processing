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

# def avg_num_nonwords(nlp, file_path, amount=100):
#     """
#     Takes in natural language processor, filepath, and word amount
#     Counts number of nonwords by checking if words are in spaCys vocabulary
#     Returns average number of nonwords per word amount
#     """
#     doc = nlp(Path(file_path).read_text(encoding='utf-8'))
#     nonwords = 0
#     total_words = 0
#     for token in doc:
#         if token.is_alpha:
#             total_words += 1
#             if token.is_oov:
#                 nonwords += 1
#     return nonwords / total_words * amount

# def incorrectly_followed_articles(nlp, file_path):
#     """
#     Takes in a natural language processor and file path
#     counts number of articles not followed by a noun, proper noun, or adjective
#     returns the count
#     """
#     doc = nlp(Path(file_path).read_text(encoding='utf-8'))
#     articles = ["a", "an", "the"]
#     count = 0
#     for i, token in enumerate(doc):
#         if token.text.lower() in articles:
#             try:
#                 following_token = doc[i+1]
#                 if following_token.pos_ not in ["ADJ", "NOUN", "PROPN"]:
#                     count += 1
#             except IndexError:
#                 count += 1
#     return count

# def count_num_sentences_without_verbs(nlp, file_path):
#     """
#     Takes in a natural language processor and file path
#     Counts the number of sentences without verbs.
#     Returns the count of sentences without verbs.
#     """
#     doc = nlp(Path(file_path).read_text(encoding='utf-8'))
#     count = 0
#     for sentence in doc.sents:
#         count_verbs =  0
#         for token in sentence:
#             if token.pos_ == "VERB":
#                 count_verbs += 1
#         if count_verbs < 1:
#             count += 1
#     return count
