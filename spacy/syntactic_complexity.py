"""
syntactic_complexity.py
Functions related to syntactic complexity.
"""

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

def tense_inflected_verbs(nlp_util, amount=100):
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
    data = get_tiv_data(nlp_util.doc, amount)
    function = 'tense_inflected_verbs'
    parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath,
        'amount': amount, 'function': function}
    final_data = {'parameters': parameters, 'data': data}
    nlp_util.write_json_model(function, final_data)

def dependency_distance(nlp_util):
    """
    Use the TextDescriptives pipeline component to get dependency distance metrics.
    dependency_distance dict data:
        dict: Dictionary with the following keys:
            - dependency_distance_mean: Mean dependency distance on the sentence
              level
            - dependency_distance_std: Standard deviation of dependency distance on
              the sentence level
            - prop_adjacent_dependency_relation_mean: Mean proportion of adjacent
              dependency relations on the sentence level
            - prop_adjacent_dependency_relation_std: Standard deviation of
              the proportion of adjacent dependency relations on the sentence level
    """
    function = 'dependency_distance'
    parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath, 'function': function}
    final_data = {'parameters': parameters, 'data': nlp_util.doc._.dependency_distance}
    nlp_util.write_json_model(function, final_data)
