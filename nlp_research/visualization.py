#table of pos for first five sentences
import spacy
import pathlib


nlp = spacy.load("en_core_web_sm")
file_name = "test.txt"
doc = nlp(pathlib.Path(file_name).read_text(encoding="utf-8"))

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)


#stop words - usefull to take out when examining word frequencies
for token in doc:
    if not token.is_stop:
        print(token.text)

#word frequencies
word_frequencies = {}
for token in doc:
    if not token.is_stop and not token.is_punct:
        if token.text not in word_frequencies:
            word_frequencies[token.text] = 1
        else:
            word_frequencies[token.text] += 1
print(word_frequencies)



