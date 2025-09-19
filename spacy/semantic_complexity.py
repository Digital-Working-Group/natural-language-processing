"""
semantic_complexity.py
Functions related to semantic complexity.
"""
import utility as util

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

def idea_density_sentences(model, filepath):
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
    doc, path_filepath = util.get_doc_and_filepath(model, filepath)
    function = 'idea_density_sentences'
    pos_list = get_prop_pos()
    sent_list = []
    for sent_idx, sent in enumerate(doc.sents):
        sent_list.append(get_idea_density_sent_data(sent_idx, sent, pos_list))
    parameters = {'model': model, 'filepath': filepath, 'function': function}
    data = {'total_sentences': len(sent_list), 'sent_list': sent_list}
    final_data = {'parameters': parameters, 'data': data}
    util.write_json_model(path_filepath, function, model, final_data)

# def generate_noun_feature(model, filepath, **kwargs):
#     """
#     Takes in a natural language processor, file path, and feature index
#     Calculates value of feature for each noun based on feature index and corresponding dataset
#     Returns average feature value across all nouns

#     model: spaCy model to load
#     filepath: text file to process

#     """
#     feature_column = kwargs['feature_column']
#     dataset_fp = kwargs['dataset_fp']
#     read_excel_kwargs = kwargs.get('read_excel_kwargs', {})
#     word_column = kwargs.get('word_column')
#     increment_result = kwargs.get('increment_result', lambda r: r.item())

#     doc, path_filepath = util.get_doc_and_filepath(model, filepath)
#     df = pd.read_excel(dataset_fp, **read_excel_kwargs)
#     df[word_column] = df[word_column].str.lower()

#     total_nouns = 0
#     total_feature_values = 0

#     for token in doc:
#         if token.pos_ == "NOUN":
#             total_nouns += 1
#             word = token.text.lower()
#             result = df.loc[df[word_column] == word, feature_column]
#             word_lemma = token.lemma_.lower()
#             result_lemma = df.loc[df[word_column] == word_lemma, feature_column]
#             if not result.empty:
#                 total_feature_values += increment_result(result)
#             elif not result_lemma.empty:
#                 total_feature_values += increment_result(result_lemma)
#             else:
#                 total_nouns -= 1

#     return total_feature_values / total_nouns

# def abstractness(nlp, file_path):
#     """
#     Uses generate_noun_feature() to calculate average abstractness value across all nouns
#     """
#     kwargs = {'feature_column': 'Conc.M', 'dataset_fp': 'datasets/dataset_for_abstractness.xlsx', 'word_column': 'Word',
#         'increment_result': lambda r: (1 / r.item())}
#     return generate_noun_feature(nlp, file_path, **kwargs)

# def semantic_ambiguity(nlp, file_path):
#     """
#     Uses generate_noun_feature() to calculate average semantic ambiguity value across all nouns
#     """
#     kwargs = {'feature_column': 'SemD', 'dataset_fp': 'datasets/dataset_for_semantic_ambiguity.xlsx', 'read_excel_kwargs': {'skiprows': 1},
#         'word_column': '!term'}
#     return generate_noun_feature(nlp, file_path, **kwargs)

# def word_frequency(nlp, file_path):
#     """
#     Uses generate_noun_feature() to calculate average word frequency value across all nouns
#     """
#     kwargs = {'feature_column': 'Lg10WF', 'dataset_fp': 'datasets/dataset_for_word_frequency.xlsx', 'word_column': 'Word'}
#     return generate_noun_feature(nlp, file_path, **kwargs)

# def word_prevalence(nlp, file_path):
#     """
#     Uses generate_noun_feature() to calculate average word prevalence value across all nouns
#     """
#     kwargs = {'feature_column': 'Prevalence', 'dataset_fp': 'datasets/dataset_for_word_prevalence_and_familiarity.xlsx', 'word_column': 'Word' }
#     return generate_noun_feature(nlp, file_path, **kwargs)

# def word_familiarity(nlp, file_path):
#     """
#     Uses generate_noun_feature() to calculate average word familiarity value across all nouns
#     """
#     kwargs = {'feature_column': 'Pknown', 'dataset_fp': 'datasets/dataset_for_word_prevalence_and_familiarity.xlsx',
#               'word_column': 'Word'}
#     return generate_noun_feature(nlp, file_path, **kwargs)

# def age_of_acquisition(nlp, file_path):
#     """
#     Uses generate_noun_feature() to calculate average age of acquisition value across all nouns
#     """
#     kwargs = {'feature_column': 'AoA', 'dataset_fp': 'datasets/dataset_for_age_of_acquisition.xlsx', 'word_column': 'Word'}
#     return generate_noun_feature(nlp, file_path, **kwargs)
