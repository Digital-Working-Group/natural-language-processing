import spacy
import pathlib
import pandas as pd

nlp = spacy.load("en_core_web_sm")
file_name = "test.txt"
doc = nlp(pathlib.Path(file_name).read_text(encoding="utf-8"))

d = []
for token in doc:
    d.append({
        "TEXT": token.text,
        "LEMMA": token.lemma_,
        "POS": token.pos_,
        "TAG": token.tag_,
        "DEP": token.dep_,
        "SHAPE": token.shape_,
        "ALPHA": token.is_alpha,
        "STOP": token.is_stop
        })

df = pd.DataFrame(data=d)
print(df.head())

