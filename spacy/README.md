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
## Data Dictionary
See the data_dictionary.md file for detailed descriptions of each linguistic feature, the function(s) used to generate them, and references to the papers they were extracted from.

## Jupyter Notebook Examples
Please run [jupyter noteboook](https://docs.jupyter.org/en/latest/running.html) and see nlp_function_examples.ipynb for an interactive set of examples. Also, see the usage example sections below.

## Extracting Linguistic Features
See `extract_linguistic_features.main()`for usage examples. 
### Parts-of-Speech Tagging
#### `data_to_df()`
The `data_to_df()` function in `pos_tagging.py` takes in a natural language processor, and a file path. Spacy converts the raw text to a Doc object that consists of tokens with various attributes. The function returns a pandas dataframe of these attributes. 

#### Parameters for `data_to_df()`
The following table shows the functions parameters and their descriptions:

| Parameter | Type                | Description                                                                                                                                                                                                                                                                                                                                                                                                                           | Default Value |
|-----------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| nlp       | spacy.lang.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their pipeline. The type can be `core`, a general purpose pipeline, or `dep` which is only for tagging, parsing, and lemmatization. The genre specifies the type of text the pipeline is trained on `web` or `news`. The size options include `sm`, `md`, `lg`, and `trf`. For all usage examples, we use spacy.load('en_core_web_lg'). | N/A           |
| file_path | str                 | This is a path to a file in string format.                                                                                                                                                                                                                                                                                                                                                                                            | N/A           |

#### `pos_tag_counts()`
The `pos_tag_counts()` function in `pos_tagging.py` takes in a natural language processor, filepath, tag("POS", or " TAG"), and per word amount. It returns a dictionary containing all tags and on average how many times they appear per 100 words
#### Parameters for `pos_tag_counts()`
The following table shows the functions parameters and their descriptions:

| Parameter | Type                    | Description                                                                                                           | Default Value |
|-----------|-------------------------|-----------------------------------------------------------------------------------------------------------------------|---------------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model.            | N/A           |
| file_path | str                     | This is a filepath in string format.                                                                                  | N/A           |
| tag       | str                     | This is either "POS", or "TAG" depending on what tags should be used.                                                 | N/A           |
| amount    | int                     | This is an integer representing the number of words for which the proportion of parts-of-speech should be calculated. | 100           |

#### `tag_ratio()`
The `tag_ratio()` function in `pos_tagging.py` takes in a natural language processor and file path as before. It additionally takes in a word amount. The function outputs a dictionary containing parts-of-speech, Penn Treebank tags, and the average number of each present per desired word amount. 

### Parameters for `tag_ratio()`

| Parameter | Type                    | Description                                                                                                          | Default Value |
|-----------|-------------------------|----------------------------------------------------------------------------------------------------------------------|---------------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model.           | N/A           |
| file_path | str                     | This is a path to a file in string format                                                                            | N/A           |
| amount    | int                     | This is an integer representing the number of words for which the proportion of parts-of-speech should be calculated | 100           |

### Semantic Complexity
#### `calculate_idea_density()`
The function `calculate_idea_density()` in `semantic_complexity.py` takes in a nlp, and file_path. The function outputs a list of the sentences in the document and their idea densities. Idea Density is defined as the ratio of propositions to total words.
#### Parameters for `calculate_idea_density()`:

| Parameter | Type                    | Description                                                                                                | Default Value |
|-----------|-------------------------|------------------------------------------------------------------------------------------------------------|---------------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model. | Required      |
| file_path | str                     | This is a path to a file in string format                                                                  | Required      |

#### `abstractness()`
The function `abstractness()` in `semantic_complexity.py` calls `generate_noun_feature()` with specified kwargs. This function finds the abstractness value corresponding to each noun in the text utilizing a pre-existing dataset, and averages these values. The function outputs the average abstractness value across all nouns in the text.
#### `semantic_ambiguity()` 
The function `semantic_ambiguity()` in `semantic_complexity.py` calls `generate_noun_feature()` with specified kwargs. This function finds the semantic ambiguity value corresponding to each noun in the text utilizing a pre-existing dataset, and averages these values. The function outputs the average semantic ambiguity value across all nouns in the text.
#### `word_frequency()`
The function `word_frequency()` in `semantic_complexity.py` calls `generate_noun_feature()` with specified kwargs. This function finds the word frequency value corresponding to each noun in the text utilizing a pre-existing dataset, and averages these values. The function outputs the average word frequency value across all nouns in the text.
#### `word_prevalence`
The function `word_prevalence()` in `semantic_complexity.py` calls `generate_noun_feature()` with specified kwargs. This function finds the word prevalence value corresponding to each noun in the text utilizing a pre-existing dataset, and averages these values. The function outputs the average word prevalence value across all nouns in the text.
#### `word_familiarity()`
The function `word_familiarity()` in `semantic_complexity.py` calls `generate_noun_feature()` with specified kwargs. This function finds the word familiarity value corresponding to each noun in the text utilizing a pre-existing dataset, and averages these values. The function outputs the average word familiarity value across all nouns in the text.
#### `age_of_acquisition()` 
The function `age_of_acquisition()` in `semantic_complexity.py` calls `generate_noun_feature()` with specified kwargs. This function finds the age of acquisition value corresponding to each noun in the text utilizing a pre-existing dataset, and averages these values. The function outputs the average age of acquisition value across all nouns in the text.
#### Parameters for `generate_noun_feature()`
The functions  `abstractness()`, `semantic_ambiguity()`, `word_frequency()`, `word_prevalence()`, `word_familiarity()`, and `age_of_acquisition()` in `semantic_complexity.py`, all depend on the `generate_noun_feature()` function. They each calculate a feature value of nouns in the text. These values estimate the complexity of the text. Each function uses a different dataset and column with predetermined values for each word. The functions all take in a natural language processor, file path, and function specific arguments via **kwargs, and return a singular float value, representing the average feature value across all nouns in the text.

| Parameter | Type                    | Description                                                                                               | Default |
|-----------|-------------------------|-----------------------------------------------------------------------------------------------------------|---------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from SpaCy. The user can choose the type, genre, and size of their model | N/A     |
| file_path | str                     | This is a path to a file in string format                                                                 | N/A     |
| **kwargs  |                         | function specific parameters for dataset lookup                                                           |         |

#### **kwargs for `generate_noun_feature()`

| Kwarg             | Type | Description                                                                         | Default            |
|-------------------|------|-------------------------------------------------------------------------------------|--------------------|
| feature_column    | str  | refers to the column in the dataset used to calculate feature                       |                    |
| dataset_fp        | str  | string version of the path to the dataset used to calculate features                |                    |
| read_excel_kwargs | dict | parameter to pass to pd.read_excel, for example {'skiprows': 1}                     | {}                 |
| word_column       | str  | refers to the name of the column used to find the corresponding word in the dataset |                    |
| increment results |      | function applied to row in dataset to add the item in the feature column            | lambda r: r.item() |

### Syntactic Complexity
#### `num_tense_inflected_verbs()`
The `num_tense_inflected_verbs()` function in `syntactic_complexity.py` takes in a nlp, file_path, and amount. The function outputs the average number of tense-inflected verbs per specified word amount. Tense inflected verbs are defined as present and past verbs and modal auxiliaries. 

### Syntactic Errors
#### `nonword_frequency()`
The `nonword_frequency()` function takes in a natural language processor, file path, dataset of words, and per word amount(default is 100). The dataset used in examples comes from [kaggle](https://www.kaggle.com/datasets/bwandowando/479k-english-words). It is text file containing over 466k English words. The function calculates frequency of non-words using the dataset and Outputs on average how many non-words are present per word amount.
#### Parameters for `nonword_frequency()`:

| Parameter  | Type                    | Description                                                                                                   | Default  |
|------------|-------------------------|---------------------------------------------------------------------------------------------------------------|----------|
| nlp        | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model.    | Required |
| file_path  | str                     | This is a filepath in string format.                                                                          | Required |
| dataset_fp | str                     | This is a filepath in string format for a dataset of English words. (used to detect nonwords)                 | Required |
| amount     | int                     | This is an integer representing the number of words for which the proportion of nonwords should be calculated | 100      |

#### `sentence_lengths()`
The `sentence_lengths()` function takes in a natural language processor and a filepath. It calculates the length of each sentence and returns a list of sentence lengths.
#### Parameters for `sentence_lengths():`

| Parameter | Type                    | Description                                                                                                | Default  |
|-----------|-------------------------|------------------------------------------------------------------------------------------------------------|----------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model. | Required |
| file_path | str                     | This is a filepath in string format.                                                                       | Required |

### Lexical Repetition
#### `most_frequent_word()`
The function `most_frequent_word()` in `lexical_repetition.py` takes in a natural language processor and a filepath. It calculates and returns the most commonly occurring word and how many times it appears in the text.
#### Parameters for `most_frequent_word()`

| Parameter  | Type                    | Description                                                                                                | Default  |
|------------|-------------------------|------------------------------------------------------------------------------------------------------------|----------|
| nlp        | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model. | Required |
| file_path  | str                     | This is a filepath in string format.                                                                       | Required |


## SpaCy Universal Tag Explanation
Please see spacy_pos_tags_explained.md for details on spaCy's parts-of-speech tags and their descriptions.

## Usage Examples
This section demonstrate how to call the functions described above.
### Parts-of-Speech Tagging
#### `data_to_df()`
```doctest
import spacy
from nlp_functions import data_to_df

data_to_df(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
#### `pos_tag_counts()`
```doctest
import spacy
from pos_tagging import pos_tag_counts

pos_tag_counts(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", tag="POS", amount=100)
pos_tag_counts(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", tag="TAG", amount=100)
```
```doctest
defaultdict(<class 'int'>, {'PRON': 13.675213675213676, 'VERB': 13.96011396011396, 'PUNCT': 12.678062678062679, 'SCONJ': 2.1367521367521367, 'ADV': 4.843304843304843, 'ADP': 8.547008547008547, 'NOUN': 17.663817663817664, 'ADJ': 6.41025641025641, 'DET': 7.4074074074074066, 'PROPN': 1.4245014245014245, 'AUX': 3.9886039886039883, 'PART': 1.9943019943019942, 'CCONJ': 4.273504273504273, 'NUM': 0.14245014245014245, 'SPACE': 0.8547008547008548})
defaultdict(<class 'int'>, {'PRP': 8.547008547008547, 'VBP': 1.7094017094017095, ',': 6.1253561253561255, 'WRB': 1.566951566951567, 'RB': 5.413105413105413, 'IN': 8.547008547008547, 'PRP$': 1.9943019943019942, 'NN': 13.96011396011396, 'VBZ': 1.7094017094017095, 'JJ': 5.555555555555555, 'DT': 8.262108262108262, 'NNS': 4.700854700854701, 'VB': 2.564102564102564, 'WP': 0.5698005698005698, '.': 4.843304843304843, 'VBD': 7.4074074074074066, 'RP': 0.5698005698005698, 'VBN': 0.7122507122507122, 'NNP': 1.4245014245014245, ':': 1.282051282051282, 'VBG': 2.849002849002849, 'CC': 4.273504273504273, 'CD': 0.14245014245014245, 'HYPH': 0.2849002849002849, '_SP': 0.8547008547008548, 'NFP': 0.14245014245014245, 'WDT': 0.5698005698005698, 'RBR': 0.2849002849002849, 'TO': 0.9971509971509971, 'JJS': 0.2849002849002849, 'POS': 0.14245014245014245, 'MD': 0.9971509971509971, 'JJR': 0.5698005698005698, 'EX': 0.14245014245014245})

```
#### `tag_ratio()`
```doctest
import spacy
from pos_tagging import tag_ratio

tag_ratio(nlp=spacy.load('en_core_web_sm'), file_path="sample_text/test.txt", amount=100)
```
```doctest
{'POS': defaultdict(<class 'int'>, {'PRON': 15.151515151515152, 'VERB': 13.636363636363635, 'PUNCT': 12.121212121212121, 'SCONJ': 4.545454545454546, 'ADV': 7.575757575757576, 'ADP': 9.090909090909092, 'NOUN': 15.151515151515152, 'AUX': 3.0303030303030303, 'ADJ': 6.0606060606060606, 'DET': 7.575757575757576, 'PROPN': 1.5151515151515151, 'PART': 1.5151515151515151, 'CCONJ': 1.5151515151515151, 'NUM': 1.5151515151515151}), 'TAG': defaultdict(<class 'int'>, {'PRP': 9.090909090909092, 'VBP': 4.545454545454546, ',': 7.575757575757576, 'WRB': 4.545454545454546, 'RB': 9.090909090909092, 'IN': 7.575757575757576, 'PRP$': 3.0303030303030303, 'NN': 12.121212121212121, 'VBZ': 1.5151515151515151, 'JJ': 6.0606060606060606, 'DT': 7.575757575757576, 'NNS': 4.545454545454546, 'VB': 1.5151515151515151, 'WP': 1.5151515151515151, '.': 4.545454545454546, 'VBD': 6.0606060606060606, 'RP': 1.5151515151515151, 'VBN': 1.5151515151515151, 'NNP': 1.5151515151515151, 'VBG': 1.5151515151515151, 'CC': 1.5151515151515151, 'CD': 1.5151515151515151})}
```
### Semantic Complexity
#### `calculate_idea_density()`
```doctest
import spacy
from semantic_complexity import calculate_idea_density

calculate_idea_density(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```json
[("You know, when I think back on my life, it's funny how the little things really shape who you become.", 0.55), ('I grew up in this small town called Ridgewood, tucked away in the countryside.', 0.5714285714285714), ("It wasn't much just rolling hills, a couple of farms, and one main street with a diner where everyone knew your name.", 0.4090909090909091)]
```
#### `abstractness()`
```doctest
import spacy
from semantic_complexity import abstractness

abstractness(nlp=spacy.load('en_core_web_lg'), file_path='sample_text/test.txt')
```
```doctest
0.29922490830321297
```

#### `semantic_ambiguity()`
```doctest
import spacy
from semantic_complexity import semantic_ambiguity

semantic_ambiguity(nlp=spacy.load('en_core_web_lg'), file_path='sample_text/test.txt')
```
```doctest
1.822231297208903
```

#### `word_frequency()`
```doctest
import spacy
from semantic_complexity import word_frequency

word_frequency(nlp=spacy.load('en_core_web_lg'), file_path='sample_text/test.txt')

```
```doctest
3.4619091967314177
```

#### `word_prevalence()`
```doctest
import spacy
from semantic_complexity import word_prevalence

print(word_prevalence(nlp=spacy.load('en_core_web_lg'), file_path='sample_text/test.txt'))
```
```doctest
2.3736957410124493
```

#### `word_familiarity()`
```doctest
import spacy
from semantic_complexity import word_familiarity

word_familiarity(nlp=spacy.load('en_core_web_lg'), file_path='sample_text/test.txt')
```
```doctest
0.9956890179911594
```

#### `age_of_acquisition()`
```doctest
import spacy
from semantic_complexity import age_of_acquisition

age_of_acquisition(sapcy.load('en_core_web_lg'), file_path='sample_text/test.txt')
```
```doctest
6.383932762273736
```
### Syntactic Complexity

#### `num_tense_inflected_verbs()`
```doctest
import spacy
from syntactic_complexity import num_tense_inflected_verbs

num_tense_inflected_verbs(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", amount=100)
```
```json
14.285714285714285
```
#### `sentence_lengths()`
```doctest
import spacy
from syntactic_complexity import sentence_lengths

sentence_lengths(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
[20, 14, 22, 19, 16, 3, 11, 22, 16, 6, 27, 9, 21, 8, 8, 25, 12, 13, 22, 18, 20, 22, 25, 5, 21, 16, 25, 9, 25, 20, 26, 21, 17, 22]

```

### Syntactic Errors
`nonword_frequency()`
```doctest
import spacy
from syntactic_errors import nonword_frequency

nonword_frequency(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", dataset_fp="words_alpha.txt", amount=100)
```
```doctest
1.0186757215619695
```

### Lexical Repetition
```doctest
import spacy
from lexical_repetition import most_frequent_word

most_frequent_word(nlp=spacy.load('en_ccore_web_lg'), file_path="sample_text/test.txt")
```
```doctest
('life', 5)
```

## Sample Input Files
The folder sample_text contains text documents for convenient testing of these NLP functions.

| File name             | Description                                                                                         | Use                                                                                                                                             |
|-----------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| test.txt              | This is a text file containing an AI generated story of 500 words.                                  | Can be used to test any of the nlp functions.                                                                                                   |
| contains_nonwords.txt | This is a modified version of test.txt, in which nonsensical words were added to several sentences. | This can be used to test functions detecting presence of non-words.                                                                             |
| sample.txt            | This is a subset of test.txt containing the first three sentences of the document.                  | This file is used in most of the Jupyter Notebook examples for quick testing and easy visualization. I is also used to validate some functions. |
