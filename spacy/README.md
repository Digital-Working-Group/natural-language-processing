# Linguistic Features: SpaCy
This repository contains scripts that show examples of how to use the [spaCy Python library](https://spacy.io/usage/linguistic-features) to generate linguistic features from raw text input.

## Installation
### Without Docker
Check your python version:
```doctest
python --version
```
Install requirements for python 3.13.5:
```
pip install -r python3-15-5_requirements.txt
```

## Jupyter Notebook Examples
Please run [jupyter noteboook](https://docs.jupyter.org/en/latest/running.html) and see nlp_function_examples.ipynb for an interactive set of examples. Also, see the usage example sections below.

## Extracting Linguistic Features
See `extract_linguistic_features.main()`for usage examples. The `data_to_df()` function takes in a natural language processor, and a file path. Spacy converts the raw text to a Doc object that consists of tokens with various attributes. The function returns a pandas dataframe of these attributes. The following table shows the functions parameters and their descriptions:

| Parameter | Type                | Description                                                                                                                                                                                                                                                                                                                                                                                                                           | Default Value |
|-----------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| nlp       | spacy.lang.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their pipeline. The type can be `core`, a general purpose pipeline, or `dep` which is only for tagging, parsing, and lemmatization. The genre specifies the type of text the pipeline is trained on `web` or `news`. The size options include `sm`, `md`, `lg`, and `trf`. For all usage examples, we use spacy.load('en_core_web_lg'). | None          |
| file_path | str                 | This is a path to a file in string format.                                                                                                                                                                                                                                                                                                                                                                                            | None          |

The `tag_ratio()` function takes in a natural language processor and file path as before. It additionally takes in a word amount. The function outputs a dictionary containing parts-of-speech and the average number of each present per desired word amount. The following table deatils the parameters

| Parameter | Type                    | Description                                                                                                          | Default Value |
|-----------|-------------------------|----------------------------------------------------------------------------------------------------------------------|---------------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model.           | None          |
| file_path | str                     | This is a path to a file in string format                                                                            | None          |
| amount    | int                     | This is an integer representing the number of words for which the proportion of parts-of-speech should be calculated | 100           |

`num_tense_inflected_verbs()` takes in a nlp, file_path, and amount as detailed above. The function outputs the average number of tense-inflected verbs per specified word amount. Tense inflected verbs are defined as present and past verbs and modal auxiliaries.

`calculate_idea_density()` takes in a nlp, and file_path as detailed above. The function outputs a list of the sentences in the document and their idea densities. Idea Density is defined as the ratio of propositions to total words.

## Usage Examples
Thi section demonstrate how to call the functions described above.
### `data_to_df()`
```doctest
import spacy
from nlp_functions import data_to_df

data_to_df(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/sample.txt")
```
### `tag_ratio()`
```doctest
import spacy
from nlp_functions import tag_ratio

tag_ratio(nlp=spacy.load('en_core_web_sm'), file_path="sample_text/sample.txt", amount=100)
```
```json
{'POS': defaultdict(<class 'int'>, {'PRON': 15.151515151515152, 'VERB': 13.636363636363635, 'PUNCT': 12.121212121212121, 'SCONJ': 4.545454545454546, 'ADV': 7.575757575757576, 'ADP': 9.090909090909092, 'NOUN': 15.151515151515152, 'AUX': 3.0303030303030303, 'ADJ': 6.0606060606060606, 'DET': 7.575757575757576, 'PROPN': 1.5151515151515151, 'PART': 1.5151515151515151, 'CCONJ': 1.5151515151515151, 'NUM': 1.5151515151515151}), 'TAG': defaultdict(<class 'int'>, {'PRP': 9.090909090909092, 'VBP': 4.545454545454546, ',': 7.575757575757576, 'WRB': 4.545454545454546, 'RB': 9.090909090909092, 'IN': 7.575757575757576, 'PRP$': 3.0303030303030303, 'NN': 12.121212121212121, 'VBZ': 1.5151515151515151, 'JJ': 6.0606060606060606, 'DT': 7.575757575757576, 'NNS': 4.545454545454546, 'VB': 1.5151515151515151, 'WP': 1.5151515151515151, '.': 4.545454545454546, 'VBD': 6.0606060606060606, 'RP': 1.5151515151515151, 'VBN': 1.5151515151515151, 'NNP': 1.5151515151515151, 'VBG': 1.5151515151515151, 'CC': 1.5151515151515151, 'CD': 1.5151515151515151})}
```

### `num_tense_inflected_verbs()`
```doctest
import spacy
from nlp_functions import num_tense_inflected_verbs

num_tense_inflected_verbs(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/sample.txt", amount=100)
```
```json
14.285714285714285
```
### `calculate_idea_density()`
```doctest
import spacy
from nlp_functions import calculate_idea_density

calculate_idea_density(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/sample.txt")
```
```json
[("You know, when I think back on my life, it's funny how the little things really shape who you become.", 0.55), ('I grew up in this small town called Ridgewood, tucked away in the countryside.', 0.5714285714285714), ("It wasn't much just rolling hills, a couple of farms, and one main street with a diner where everyone knew your name.", 0.4090909090909091)]
```

## Sample Input Files
The folder sample_text contains text documents for convenient testing of these NLP functions.

| File name             | Description                                                                                         | Use                                                                                                                                             |
|-----------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| test.txt              | This is a text file containing an AI generated story of 500 words.                                  | Can be used to test any of the nlp functions.                                                                                                   |
| contains_nonwords.txt | This is a modified version of test.txt, in which nonsensical words were added to several sentences. | This can be used to test functions detecting presence of non-words.                                                                             |
| sample.txt            | This is a subset of test.txt containing the first three sentences of the document.                  | This file is used in most of the Jupyter Notebook examples for quick testing and easy visualization. I is also used to validate some functions. |
