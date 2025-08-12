from nlp_functions import data_to_df, tag_ratio, num_tense_inflected_verbs, calculate_idea_density
import spacy

def main():

    """
    Main entry point for extracting linguistic features.
    """
    #tokens and attributes visualized in a dataframe
    print(data_to_df(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt").head())

    #parts-of-speech tagging
    print(tag_ratio(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", amount=100))

    #number of tense inflected verbs per 100 words
    print(num_tense_inflected_verbs(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", amount=100))

    #sentences and their idea density
    print(calculate_idea_density(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt"))

if __name__ == '__main__':
    main()