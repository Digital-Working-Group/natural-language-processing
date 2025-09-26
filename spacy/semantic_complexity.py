"""
semantic_complexity.py
Functions related to semantic complexity.
"""
import pandas as pd

def get_prop_pos():
    """
    get prop pos tags for idea_density_sentences()
    """
    verb_pos = ['VERB']
    adj_pos = ['ADJ']
    adv_pos = ['ADV']
    prep_pos = ['ADP']
    conj_pos = ['CONJ', 'CCONJ', 'SCONJ']
    return verb_pos + adj_pos + adv_pos + prep_pos + conj_pos

def get_idea_density_sent_data(sent_idx, sent, pos_list):
    """
    get sentence data for idea_density_sentences()
    """
    props = 0
    total_tokens = 0
    for token in sent:
        if token.is_alpha:
            total_tokens += 1
            if token.pos_ in pos_list:
                props += 1
    return {'sent_idx': sent_idx, 'total_tokens': total_tokens,
        'idea_density': props / total_tokens}

def idea_density_sentences(nlp_util):
    """
    Examines only alphanumeric (is_alpha) characters
    
    Calculates idea density per sentence:
        the number of propositions divided by the total words in the sentence
    Propositions include verbs, adjectives, adverbs, prepositions, and conjunctions.
        VERB: verbs
        ADJ: adjectives
        ADV: adverbs
        ADP: adposition (prepositions and postpositions)
        CONJ: conjunction
        CCONJ: coordinating conjunction
        SCONJ: subordinating conjunction
    Idea density is also known as propositional density (or P-density)
    Details about the spaCy POS tags  in spacy_pos_tags_explained.md
    
    model: spaCy model to load
    filepath: text file to process
    """
    function = 'idea_density_sentences'
    pos_list = get_prop_pos()
    sent_list = []
    for sent_idx, sent in enumerate(nlp_util.doc.sents):
        sent_list.append(get_idea_density_sent_data(sent_idx, sent, pos_list))
    parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath, 'function': function}
    data = {'total_sentences': len(sent_list), 'sent_list': sent_list}
    final_data = {'parameters': parameters, 'data': data}
    nlp_util.write_json_model(function, final_data)

def process_doc_noun_feature(doc, **kwargs):
    """
    process a doc for generate_noun_feature()
    """
    dataset_fp = kwargs['dataset_fp']
    read_excel_kwargs = kwargs.get('read_excel_kwargs', {})
    word_column = kwargs.get('word_column')
    feature_column = kwargs['feature_column']
    increment_result = kwargs.get('increment_result', lambda r: r.item())
    df = pd.read_excel(dataset_fp, **read_excel_kwargs)
    df[word_column] = df[word_column].str.lower()
    feature_data = []
    for token in doc:
        if token.pos_ == "NOUN":
            word = token.text.lower()
            result = df.loc[df[word_column] == word, feature_column]
            word_lemma = token.lemma_.lower()
            result_lemma = df.loc[df[word_column] == word_lemma, feature_column]
            if not result.empty:
                feature_data.append({'token.text': token.text, 'word': word,
                    'feature_val': increment_result(result), 'is_lemma': 0})
            elif not result_lemma.empty:
                feature_data.append({'token.text': token.text, 'word': word_lemma,
                    'feature_val': increment_result(result_lemma), 'is_lemma': 1})
    return feature_data

def generate_noun_feature(nlp_util, **kwargs):
    """
    Calculates value of feature for each noun based on feature index and corresponding dataset
    Returns the average feature value across all nouns
    If the noun isn't found in the corresponding dataset, it is ignored and not included in the
    total number of nouns.

    model: spaCy model to load
    filepath: text file to process

    """
    feature = kwargs['feature']
    feature_data = process_doc_noun_feature(nlp_util.doc, **kwargs)
    parameters = {'model': nlp_util.model, 'filepath': nlp_util.filepath}
    parameters.update({k: v for k, v in kwargs.items() if k != 'increment_result'})
    final_data = {'parameters': parameters,
        'data': {'total_nouns': len(feature_data), 'feature_data': feature_data}}
    nlp_util.write_json_model(feature, final_data)

def abstractness(nlp_util):
    """
    Uses generate_noun_feature() to calculate average abstractness value across all nouns
    """
    kwargs = {'feature_column': 'Conc.M',
        'dataset_fp': 'datasets/dataset_for_abstractness.xlsx', 'word_column': 'Word',
        'increment_result': lambda r: (1 / r.item()), 'feature': 'abstractness'}
    generate_noun_feature(nlp_util, **kwargs)

def age_of_acquisition(nlp_util):
    """
    Uses generate_noun_feature() to calculate average age of acquisition value across all nouns
    """
    kwargs = {'feature_column': 'AoA',
        'dataset_fp': 'datasets/dataset_for_age_of_acquisition.xlsx',
        'word_column': 'Word', 'feature': 'age_of_acquisition'}
    generate_noun_feature(nlp_util, **kwargs)

def semantic_ambiguity(nlp_util):
    """
    Uses generate_noun_feature() to calculate average semantic ambiguity value across all nouns
    """
    kwargs = {'feature_column': 'SemD',
        'dataset_fp': 'datasets/dataset_for_semantic_ambiguity.xlsx',
        'read_excel_kwargs': {'skiprows': 1}, 'word_column': '!term',
        'feature': 'semantic_ambiguity'}
    generate_noun_feature(nlp_util, **kwargs)

def word_familiarity(nlp_util):
    """
    Uses generate_noun_feature() to calculate average word familiarity value across all nouns
    """
    kwargs = {'feature_column': 'Pknown',
            'dataset_fp': 'datasets/dataset_for_word_prevalence_and_familiarity.xlsx',
            'word_column': 'Word', 'feature': 'word_familiarity'}
    generate_noun_feature(nlp_util, **kwargs)

def word_frequency(nlp_util):
    """
    Uses generate_noun_feature() to calculate average word frequency value across all nouns
    """
    kwargs = {'feature_column': 'Lg10WF', 'dataset_fp': 'datasets/dataset_for_word_frequency.xlsx',
        'word_column': 'Word', 'feature': 'word_frequency'}
    generate_noun_feature(nlp_util, **kwargs)

def word_prevalence(nlp_util):
    """
    Uses generate_noun_feature() to calculate average word prevalence value across all nouns
    """
    kwargs = {'feature_column': 'Prevalence',
        'dataset_fp': 'datasets/dataset_for_word_prevalence_and_familiarity.xlsx',
        'word_column': 'Word', 'feature': 'word_prevalence'}
    generate_noun_feature(nlp_util, **kwargs)
