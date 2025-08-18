from pathlib import Path
import spacy
import pandas as pd

def calculate_idea_density(nlp=None, file_path=None):
    """
    Takes in a natural language processor, and file path
    Filters on alphabetic tokens (words), calculates idea density - the number of propositions divided by the total words
    Returns each sentence and its idea density
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    idea_density_sentences = []
    for sent in doc.sents:
        count_props = 0
        count_words = 0
        for token in sent:
            if token.is_alpha:
                count_words += 1
                if token.pos_ in ["VERB", "ADJ", "ADV", "CCONJ", "SCONJ", "ADP"]:
                    count_props += 1
        idea_density = count_props/count_words
        sentence_and_idea_density = (sent.text, idea_density)
        idea_density_sentences.append(sentence_and_idea_density)

    return idea_density_sentences

def features_of_nouns(nlp, file_path, **kwargs):
    """
    Takes in a natural language processor, file path, and feature index
    Calculates value of feature for each noun based on feature index and corresponding dataset
    Returns average feature value across all nouns
    """
    feature_column = kwargs['feature_column']
    dataset_fp = kwargs['dataset_fp']
    read_excel_kwargs = kwargs.get('read_excel_kwargs', {})
    word_column = kwargs.get('word_column')
    increment_result = kwargs.get('increment_result', lambda r: r.item())

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    df = pd.read_excel(dataset_fp, **read_excel_kwargs)
    df[word_column] = df[word_column].str.lower()

    total_nouns = 0
    total_feature_values = 0

    for token in doc:
        if token.pos_ == "NOUN":
            total_nouns += 1
            word = token.text.lower()
            result = df.loc[df[word_column] == word, feature_column]
            word_lemma = token.lemma_.lower()
            result_lemma = df.loc[df[word_column] == word_lemma, feature_column]

            if not result.empty:
                total_feature_values += increment_result(result)
            elif not result_lemma.empty:
                total_feature_values += increment_result(result_lemma)
            else:
                total_nouns -= 1

    return total_feature_values / total_nouns

def abstractness(nlp, file_path):
    """
    Uses features_of_nouns to calculate average abstractness value across all nouns
    """
    kwargs = {'feature_column': 'Conc.M', 'dataset_fp': 'datasets/dataset_for_abstractness.xlsx', 'word_column': 'Word',
        'increment_result': lambda r: (1 / r.item())}
    return features_of_nouns(nlp, file_path, **kwargs)

def semantic_ambiguity(nlp, file_path):
    """
    Uses features_of_nouns to calculate average semantic ambiguity value across all nouns
    """
    kwargs = {'feature_column': 'SemD', 'dataset_fp': 'datasets/dataset_for_semantic_ambiguity.xlsx', 'read_excel_kwargs': {'skiprows': 1},
        'word_column': '!term'}
    return features_of_nouns(nlp, file_path, **kwargs)

def word_frequency(nlp, file_path):
    """
    Uses features_of_nouns to calculate average word frequency value across all nouns
    """
    kwargs = {'feature_column': 'Lg10WF', 'dataset_fp': 'datasets/dataset_for_word_frequency.xlsx', 'word_column': 'Word'}
    return features_of_nouns(nlp, file_path, **kwargs)

def word_prevalence(nlp, file_path):
    """
    Uses features_of_nouns to calculate average word prevalence value across all nouns
    """
    kwargs = {'feature_column': 'Prevalence', 'dataset_fp': 'datasets/dataset_for_word_prevalence_and_familiarity.xlsx', 'word_column': 'Word' }
    return features_of_nouns(nlp, file_path, **kwargs)

def word_familiarity(nlp=None, file_path=None):
    """
    Uses features_of_nouns to calculate average word familiarity value across all nouns
    """
    kwargs = {'feature_column': 'Pknown', 'dataset_fp': 'datasets/dataset_for_word_prevalence_and_familiarity.xlsx',
              'word_column': 'Word'}
    return features_of_nouns(nlp, file_path, **kwargs)

def age_of_acquisition(nlp=None, file_path=None):
    """
    Uses features_of_nouns to calculate average age of acquisition value across all nouns
    """
    kwargs = {'feature_column': 'AoA', 'dataset_fp': 'datasets/dataset_for_age_of_acquisition.xlsx', 'word_column': 'Word'}
    return features_of_nouns(nlp, file_path, **kwargs)