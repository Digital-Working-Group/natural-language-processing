import spacy
import pathlib

nlp = spacy.load("en_core_web_sm")
file_name = "test.txt"
doc = nlp(pathlib.Path(file_name).read_text(encoding="utf-8"))

sentence_list = list(doc.sents)
for sentence in sentence_list[:5]:
    print(sentence)

#pos and counts
def count_pos(doc, per_word_amount=100):
    total_words = 0
    parts_of_speech = {}

    for token in doc:
        # count total words
        if token.is_alpha:   #all chars are alphabetic
            total_words += 1
        # count pos amounts
        if token.pos_ not in parts_of_speech:
            parts_of_speech[token.pos_] = 1
        else:
            parts_of_speech[token.pos_] += 1
    #pos ammount per word amount(default = 100)
    for pos in parts_of_speech.keys():
        parts_of_speech[pos] = (parts_of_speech[pos] / total_words) * per_word_amount

    print(parts_of_speech)

count_pos(doc)


#word frequencies
word_frequencies = {}
for token in doc:
    if not token.is_stop and not token.is_punct:
        if token.text not in word_frequencies:
            word_frequencies[token.text] = 1
        else:
            word_frequencies[token.text] += 1
print(word_frequencies)

#stop words - usefull to take out when examining word frequencies
for token in doc:
    if not token.is_stop:
        print(token.text)
