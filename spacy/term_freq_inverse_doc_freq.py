import math
import spacy
from pathlib import Path

def term_frequency(nlp, file_path, term=None):
    """
    Takes in a natural language processor and file path and target string
    Returns frequency of target string in document
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    term_appearances = 0
    total_words = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
        if token.text.lower() == term:
            term_appearances += 1
    return term_appearances / total_words

def inverse_document_frequency(document_list=None, term=None):
    """
    Takes in a document and target string
    Calculates inverse document frequency by taking log of (number of documents) / (number of documents containing term)
    returns inverse document frequency value
    """
    number_of_documents = len(document_list)
    print(number_of_documents)
    documents_containing_term = 0
    for document in document_list:
        term_freq_doc = term_frequency(nlp=spacy.load('en_core_web_lg'), file_path=document, term=term)
        if term_freq_doc > 0:
            print(document)
            documents_containing_term += 1
            print(documents_containing_term)
    print(math.log10(number_of_documents / documents_containing_term))
    return math.log10(number_of_documents / documents_containing_term) if documents_containing_term > 0 else 0

def tf_idf(nlp, file_path, document_list=None, term=None):
    """
    Takes in a natural language processor, filepath, document list and target string(term)
    Calculates TF-IDF by multiplying TF by IDF
    returns TF-IDF value
    """
    return term_frequency(nlp, file_path, term) * inverse_document_frequency(document_list, term)



tf_idf(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", document_list=["sample_text/sample.txt", "sample_text/test.txt", "sample_text/contains_nonwords.txt"], term="life")






