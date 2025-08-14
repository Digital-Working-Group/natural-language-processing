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