"""
syntactic_complexity.py
Functions related to syntactic complexity.
"""
import utility as util

def get_tiv_data(doc, amount):
    """
    get data for tense_inflected_verbs()
    """
    feature_data = []
    present_verbs = 0
    past_verbs = 0
    modal_auxiliaries = 0
    total_alpha_tokens = 0
    for token in doc:
        if token.is_alpha:
            total_alpha_tokens += 1
            text = token.text
            pos = token.pos_
            tag = token.tag_
            tense = token.morph.get('Tense')
            if pos == "VERB":
                word_type = None
                if "Pres" in token.morph.get("Tense"):
                    word_type = 'present_verb'
                    present_verbs += 1
                elif "Past" in token.morph.get("Tense"):
                    word_type = 'past_verb'
                    past_verbs += 1
                if word_type is not None:
                    feature_data.append({'token.text': text,
                        'token.pos_': token.pos_, 'tense': tense,
                        'tag': tag, 'word_type': word_type})
            elif pos == "AUX" and tag == "MD":
                word_type = 'modal_auxiliary'
                feature_data.append({'token.text': text,
                    'token.pos_': token.pos_, 'tense': tense,
                    'tag': tag, 'word_type': word_type})
                modal_auxiliaries += 1
    total_tivs = len(feature_data)
    tiv_ratio = total_tivs / total_alpha_tokens * amount
    return {'tiv_ratio': tiv_ratio, 'total_tivs': total_tivs,
        'total_alpha_tokens': total_alpha_tokens, 'present_verbs': present_verbs,
        'past_verbs': past_verbs, 'modal_auxiliaries': modal_auxiliaries,
        'feature_data': feature_data}

def tense_inflected_verbs(model, filepath, amount=100):
    """
    Iterates over alphanumeric tokens (token.is_alpha) and counts only tense-inflected verbs.
        Tense-inflected verbs include present and past verbs, as well as modal auxiliary verbs.
    Tense-inflected verbs:
        Present or past verb:
            token.pos_ = VERB and token.morph.get("Tense") is in ['Pres', 'Past']
        Modal auxiliary verb:
            token.pos_ = AUX and token.tag_ = MD
    Details about the spaCy POS tags  in spacy_pos_tags_explained.md
    VERB: verb
    AUX: auxiliary
    MD: verb, modal auxiliary
    """
    doc, path_filepath = util.get_doc_and_filepath(model, filepath)
    data = get_tiv_data(doc, amount)
    function = 'tense_inflected_verbs'
    parameters = {'model': model, 'filepath': filepath, 'amount': amount, 'function': function}
    final_data = {'parameters': parameters, 'data': data}
    util.write_json_model(path_filepath, function, model, final_data)

# def sentence_lengths(nlp, file_path):
#     """
#     Takes in a natural language processor and file path
#     Calculates length of sentences and appends to list
#     Returns list of sentence lengths
#     """
#     doc = nlp(Path(file_path).read_text(encoding='utf-8'))
#     length_of_sentences = []
#     for sentence in doc.sents:
#         num_words = 0
#         for token in sentence:
#             if token.is_alpha:
#                 num_words += 1
#         length_of_sentences.append(num_words)
#     return length_of_sentences

# def tree_height(node=None):
#     """
#     Returns the max height of a tree given a node
#     """
#     if node is None:
#         return 0
#     else:
#         children = list(node.lefts) + list(node.rights)
#         if not children:
#             return 1
#         else:
#             return max(tree_height(child) for child in children) + 1

# def dependency_tree_heights(nlp, file_path):
#     """
#     Takes in a natural language processor and file path
#     Uses tree_height() to calculate max height of dependency trees
#     Returns average height of all dependency trees
#     """
#     doc = nlp(Path(file_path).read_text(encoding='utf-8'))
#     tree_depths = []
#     max_depth = 0
#     for sentence in doc.sents:
#         for token in sentence:
#             if token.dep_ == "ROOT":
#                 max_depth = tree_height(token)
#         tree_depths.append(max_depth)
#     return tree_depths
