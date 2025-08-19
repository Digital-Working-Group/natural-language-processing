from pathlib import Path

def num_tense_inflected_verbs(nlp=None, file_path=None, amount=100):
    """
    Takes in a natural language processor, file path, and per word amount(default is 100)
    Filters on tokens that only contain alphabetic characters (excluding punctuation or numbers), and calculates avg number of tense inflected verbs per word amount
    Tokens with the pos: "AUX" are auxiliaries, and those with the tag: "MD" are modal auxiliaries
    Returns the number of tense inflected verbs
    """

    doc = nlp(Path(file_path).read_text(encoding='utf-8'))

    total_words = 0
    present_verbs = 0
    past_verbs = 0
    modal_auxiliary = 0
    num_tiv = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            #find tense inflected verbs
            if token.pos_ == "VERB" and "Pres" in token.morph.get("Tense"):
                present_verbs += 1
            if token.pos_ == "VERB" and "Past" in token.morph.get("Tense"):
                past_verbs += 1
            if token.pos_ == "AUX" and token.tag_ == "MD":
                modal_auxiliary += 1

            num_tiv = present_verbs + modal_auxiliary + past_verbs
    return num_tiv / total_words * amount

def sentence_lengths(nlp, file_path):
    """
    Takes in a natural language processor and file path
    Calculates length of sentences and appends to list
    Returns list of sentence lengths
    """
    doc = nlp(Path(file_path).read_text(encoding='utf-8'))
    length_of_sentences = []
    for sentence in doc.sents:
        num_words = 0
        for token in sentence:
            if token.is_alpha:
                num_words += 1
        length_of_sentences.append(num_words)
    return length_of_sentences