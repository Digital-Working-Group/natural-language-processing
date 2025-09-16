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
Please run [jupyter notebook](https://docs.jupyter.org/en/latest/running.html) and see [nlp_function_examples.ipynb](nlp_function_examples.ipynb) for an interactive set of examples. Also, see the usage example sections below.

## Load spaCy models
Prior to being able to load spaCy models such as [en_core_web_lg](https://spacy.io/models/en#en_core_web_lg), one must run `spacy download <model name`:
```sh
spacy download en_core_web_lg
```

```python
nlp = spacy.load('en_core_web_lg')
```

Without downloading the model, one may encounter errors such as:
```sh
OSError: [E050] Can't find model 'en_core_web_lg'. It doesn't seem to be a Python package or a valid path to a data directory.
```

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

#### `ratio_of_pos()`
The `ratio_of_pos()` function in `pos_tagging.py` takes in a natural language processor, filepath, key word arguments. It returns the ratio of the desired part of speech to total words in the text.
#### `ratio_of_nouns()` 
The `ratio_of_nouns()` function in `pos_tagging.py` calls ratio_of_pos() with specified kwargs to calculate and return the ratio of nouns to total words
#### `ratio_of_pronouns`
The `ratio_of_pronouns()` function in `pos_tagging.py` calls ratio_of_pos() with specified kwargs to calculate and return the ratio of pronouns to total words
#### `ratio_of_conjunctions`
The `ratio_of_conjunctions()` function in `pos_tagging.py` calls ratio_of_pos() with specified kwargs to calculate and return the ratio of conjunctions to total words
#### Parameters for `ratio_of_pos()`

| Parameter | Type                    | Description                                                                                               | Default Value |
|-----------|-------------------------|-----------------------------------------------------------------------------------------------------------|---------------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from SpaCy. The user can choose the type, genre, and size of their model | Required      |
| file_path | str                     | This is a path to a file in string format	                                                                | Required      |
| **kwargs  |                         | function specific parameters for part of speech specification                                             |               |

#### kwargs for `ratio_of_pos`

| kwarg           | Type | Description                                       | Default Value |
|-----------------|------|---------------------------------------------------|---------------|
| parts_of_speech | list | List of pos tags used to identify parts of speech | Required      |

#### `stats_proportion_part_of_speech()`
The function `stats_proportion_part_of_speech()` in `pos_tagging.py` takes in a natural language processor, file path, and set of kwargs. It calculates the ratio of specified part-of-speech to total words in a sentence for all sentences and returns the average, minimum, maximum, and standard deviation across all sentences.
#### `stats_proportion_coordinators()`
The function `stats_proportion_coordinators()` in `pos_tagging.py` takes in a natural language processor and file path. It calls stats_proportion_part_of_speech with specified kwargs to determine mean, min, max, and standard deviation of the proportion of coordinators in a sentence.
#### `stats_proportion_auxiliaries()`
The function `stats_proportion_auxiliaries()` in `pos_tagging.py` takes in a natural language processor and file path. It calls stats_proportion_part_of_speech with specified kwargs to determine mean, min, max, and standard deviation of the proportion of auxiliaries in a sentence.
#### `stats_proportion_adjectives()`
The function `stats_proportion_adjectives()` in `pos_tagging.py` takes in a natural language processor and file path. It calls stats_proportion_part_of_speech with specified kwargs to determine mean, min, max, and standard deviation of the proportion of adjectives in a sentence.
#### `stats_proportion_subjects()`
The function `stats_proportion_subjects()` in `pos_tagging.py` takes in a natural language processor and file path. It calls stats_proportion_part_of_speech with specified kwargs to determine mean, min, max, and standard deviation of the proportion of subjects in a sentence.

#### Parameters for `stats_proportion_part_of_speech()`
| Parameter | Type                    | Description                                                                                               | Default Value |
|-----------|-------------------------|-----------------------------------------------------------------------------------------------------------|---------------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from SpaCy. The user can choose the type, genre, and size of their model | Required      |
| file_path | str                     | This is a path to a file in string format	                                                                | Required      |
| **kwargs  |                         | function specific parameters for part of speech specification                                             |               |

#### kwargs for `stats_proportion_part_of_speech()`

| kwarg | Type | Description                 | Default Value |
|-------|------|-----------------------------|---------------|
| tag   | str  | parts of speech tag         | None          |
| dep   | str  | parts of speech dependency  | None          |

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
The functions  `abstractness()`, `semantic_ambiguity()`, `word_frequency()`, `word_prevalence()`, `word_familiarity()`, and `age_of_acquisition()` in `semantic_complexity.py`, all depend on the `generate_noun_feature()` function. They each calculate a feature value of nouns in the text. These values estimate the complexity of the text. Each function uses a different dataset and column with predetermined values for each word. The functions all take in a natural language processor, file path, and function specific arguments via \*\*kwargs, and return a singular float value, representing the average feature value across all nouns in the text.

| Parameter | Type                    | Description                                                                                               | Default |
|-----------|-------------------------|-----------------------------------------------------------------------------------------------------------|---------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from SpaCy. The user can choose the type, genre, and size of their model | N/A     |
| file_path | str                     | This is a path to a file in string format                                                                 | N/A     |
| **kwargs  |                         | function specific parameters for dataset lookup                                                           |         |

#### \*\*kwargs for `generate_noun_feature()`

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
#### Parameters for `num_tense_inflected_verbs()`

| Parameter | Type                    | Description                                                                                                     | Default  |
|-----------|-------------------------|-----------------------------------------------------------------------------------------------------------------|----------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model.      | Required |
| file_path | str                     | This is a filepath in string format.                                                                            | Required |
| amount    | str                     | This is an integer representing the number of words for which the proportion of nonwords should be calculated   | 100      |

#### `sentence_lengths()`
The `sentence_lengths()` function in `syntactic_complexity.py` takes in a natural language processor and a filepath. It calculates the length of each sentence and returns a list of sentence lengths.

#### `dependency_tree_heights()` 
The `dependency_tree_heights()` function in `syntactic complexity.py` takes in a natural language processor and a filepath. It calculates the dependency tree height of each dependant relation in spaCy using `tree_heights()`and returns a list of dependency tree heights. 
#### Parameters for `sentence_lengths()` and `dependency_tree_heights()`:

| Parameter | Type                    | Description                                                                                                | Default  |
|-----------|-------------------------|------------------------------------------------------------------------------------------------------------|----------|
| nlp       | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model. | Required |
| file_path | str                     | This is a filepath in string format.                                                                       | Required |

### Lexical Variation
#### `windowed_type_token_ratio()`
The `windowed_type_token_ratio()` in  function takes in a natural language processor, filepath, and window size. It calculates and returns the average type token ratio across moving windows, starting from the beginning and moving one word at a time through the text. 
#### Parameters for `windowed_type_token_ratio()`

| Parameter   | Type                    | Description                                                                                                | Default  |
|-------------|-------------------------|------------------------------------------------------------------------------------------------------------|----------|
| nlp         | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model. | Required |
| file_path   | str                     | This is a filepath in string format.                                                                       | Required |
| window_size | int                     | number of words contained in moving windows (size)                                                         | 20       |
#### `number_of_unique_tokens()`
The function `number_of_unique_tokens()` in `lexical_variation.py` takes in a natural language processor and filepath. It calculates and returns the number of unique tokens in the text. 
#### `number_of_unique_lemmas()`
The function `number_of_unique_lemmas()` in `lexical_variation.py` takes in a natural language processor and filepath. It calculates and returns the number of unique lemmas in the text. 
#### Parameters for `number_of_unique_tokens()` and `number_of_unique_lemmas()`:
| Parameter   | Type                    | Description                                                                                                | Default  |
|-------------|-------------------------|------------------------------------------------------------------------------------------------------------|----------|
| nlp         | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model. | Required |
| file_path   | str                     | This is a filepath in string format.                                                                       | Required |

### Syntactic Errors
#### `nonword_frequency()`
The `nonword_frequency()` function in `syntactic_errors.py` takes in a natural language processor, file path, dataset of words, and per word amount(default is 100). The dataset used in examples comes from [kaggle](https://www.kaggle.com/datasets/bwandowando/479k-english-words). It is text file containing over 466k English words. The function calculates frequency of non-words using the dataset and Outputs on average how many non-words are present per word amount.
#### Parameters for `nonword_frequency()`:

| Parameter  | Type                    | Description                                                                                                   | Default  |
|------------|-------------------------|---------------------------------------------------------------------------------------------------------------|----------|
| nlp        | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model.    | Required |
| file_path  | str                     | This is a filepath in string format.                                                                          | Required |
| dataset_fp | str                     | This is a filepath in string format for a dataset of English words. (used to detect nonwords)                 | Required |
| amount     | int                     | This is an integer representing the number of words for which the proportion of nonwords should be calculated | 100      |
#### `avg_num_nonwords()` 
`avg_num_nonwords()` in `syntactic_errors.py` is an alternative way of counting non-words. The function takes in a natural language processor, filepath, and word amount. It counts number of nonwords by checking if words are in spaCy's vocabulary, and returns the number of nonwords present per word amount.
#### Parameters for `avg_num_nonwords()`
| Parameter  | Type                    | Description                                                                                                   | Default  |
|------------|-------------------------|---------------------------------------------------------------------------------------------------------------|----------|
| nlp        | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model.    | Required |
| file_path  | str                     | This is a filepath in string format.                                                                          | Required |
| amount     | int                     | This is an integer representing the number of words for which the proportion of nonwords should be calculated | 100      |
#### `incorrectly_followed_articles()`
The function `incorrectly_followed_articles()` in `syntactic_errors.py` takes in a natural language processor and filepath. It calculates and returns the number of articles (a, and, the) that are not followed by an adjective, noun, or, proper noun.
#### `count_num_sentences_without_verbs()`
The function `count_num_sentences_without_verbs()` in `syntactic_errors.py` takes in a natural language processor and filepath. It calculates and returns the number of sentences in the text that do not contain verbs.
#### Parameters for `incorrectly_followed_articles()`, and `count_num_sentences_without_verbs()`:
| Parameter  | Type                    | Description                                                                                                   | Default  |
|------------|-------------------------|---------------------------------------------------------------------------------------------------------------|----------|
| nlp        | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model.    | Required |
| file_path  | str                     | This is a filepath in string format.                                                                          | Required |

### Lexical Repetition
#### `most_frequent_word()`
The function `most_frequent_word()` in `lexical_repetition.py` takes in a natural language processor and a filepath. It calculates and returns the most commonly occurring word and how many times it appears in the text.
#### `repeating_unique_word_ratio()`
The function `repeating_unique_word_ratio()` in `lexical_repetition.py` takes in a natural language processor, and a file path. It calculates and returns the ratio of repeating words to unique words in the text.
#### `total_consecutive_words()`
The function `total_consecutive_words` in `lexical_repetition.py` takes in a natural language processor and filepath. It calculates and returns the number of consecutive repeating words in the text.
#### Parameters for `most_frequent_word()`, `repeating_unique_word_ratio()`, and `total_consecutive_words()` 

| Parameter  | Type                    | Description                                                                                                | Default  |
|------------|-------------------------|------------------------------------------------------------------------------------------------------------|----------|
| nlp        | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model. | Required |
| file_path  | str                     | This is a filepath in string format.                                                                       | Required |

### Similarity
#### `stats_similarity_of_words()`
The function `stats_similarity_of_words()` in `similarity.py` takes in a natural language processor, file path, and window size(default 3). It returns dictionary containing mean, min,  max, and standard deviation of word similarity across all windows.
#### Parameters for `stats_similarity_of_words()`
| Parameter   | Type                    | Description                                                                                                | Default  |
|-------------|-------------------------|------------------------------------------------------------------------------------------------------------|----------|
| nlp         | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model. | Required |
| file_path   | str                     | This is a filepath in string format.                                                                       | Required |
| window_size | int                     | This is an integer representing the number of words contained in each window                               | 3        |
#### `mean_similarity_of_sentences()`
The function `mean_similarity_of_sentences()` in `similarity.py` takes in a natural language processor and a filepath. It calculates and returns the mean similarity of all combinations of sentences.
#### Parameters for `mean_similarity_of_sentences()`
| Parameter   | Type                    | Description                                                                                                | Default  |
|-------------|-------------------------|------------------------------------------------------------------------------------------------------------|----------|
| nlp         | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model. | Required |
| file_path   | str                     | This is a filepath in string format.                                                                       | Required |

### Term Frequency - Inverse Document Frequency
TF-IDF is a method used to evaluate importance of a word to a document in relation to a larger collection of documents. It combines term frequency (how often a word appears in a document) with inverse document frequency (total number of documents / number of documents containing term t). More information on this implementation can be found [here](https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/)
#### `tf_idf()`
The function `tf_idf()` in `term_freq_inverse_doc_freq.py` takes in  a natural language processor, filepath, document list and target string(term). It first calculates term frequency, which is defined as the frequency of a target string in a document. Then, it calculates inverse-document-frequency, which is defined as log10 of the number of documents divided by the number of documents containing the term. TF-IDF is calculated by multiplying these two values.
#### Parameters for `tf_idf()`
| Parameter     | Type                    | Description                                                                                                | Default  |
|---------------|-------------------------|------------------------------------------------------------------------------------------------------------|----------|
| nlp           | spacy.language.Language | This is a pipeline object loaded from spacy. The user can choose the type, genre, and size of their model. | Required |
| file_path     | str                     | This is a filepath in string format.                                                                       | Required |
| document_list | list                    | List of file paths                                                                                         | Required |
| term          | str                     | Target string for which TF-IDF will be calculated                                                          | None     |
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
#### `ratio_of_nouns()`
```doctest
import spacy
from pos_tagging import ratio_of_nouns

ratio_of_nouns(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
0.22866894197952217
```
#### `ratio_of_pronouns()`
```doctest
import spacy
from pos_tagging import ratio_of_pronouns

ratio_of_pronouns(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
0.16382252559726962
```
#### `ratio_of_conjunctions()`
```doctest
import spacy
from pos_tagging import ratio_of_conjunctions

ratio_of_conjunctions(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
0.025597269624573378
```
#### `stats_proportion_coordinators()`
```doctest
import spacy
from pos_tagging import stats_proportion_coordinators

stats_proportion_coordinators(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
{'mean': 0.05029893149270312, 'max': 0.1875, 'min': 0.0, 'std': 0.04583023800082182}
```
#### `stats_proportion_auxiliaries()`
```doctest
import spacy
from pos_tagging import stats_proportion_auxiliaries

stats_proportion_auxiliaries(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
{'mean': 0.05560743436130433, 'max': 0.3333333333333333, 'min': 0.0, 'std': 0.07735664309842447}
```
#### `stats_proportion_adjectives()`
```doctest
import spacy
from pos_tagging import stats_proportion_adjectives

stats_proportion_adjectives(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
{'mean': 0.09117393468581776, 'max': 0.375, 'min': 0.0, 'std': 0.09741581819031751}
```
#### `stats_proportion_subjects()`
```doctest
import spacy
from pos_tagging import stats_proportion_subjects

stats_proportion_subjects(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
{'mean': 0.13538585215010274, 'max': 0.3333333333333333, 'min': 0.04, 'std': 0.06797608583749848}
```
### Semantic Complexity
#### `calculate_idea_density()`
```doctest
import spacy
from semantic_complexity import calculate_idea_density

calculate_idea_density(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
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
#### `dependency_tree_heights()`
```doctest
import spacy
from syntactic_complexity import dependency_tree_heights, tree_heights

dependency_tree_heights(nlp=(spacy.load('en_core_web_lg')), file_path="sample_text/test.txt")
```
```doctest
[6, 5, 8, 6, 6, 2, 4, 6, 4, 3, 12, 4, 9, 5, 5, 9, 7, 5, 5, 6, 9, 5, 9, 2, 7, 8, 7, 4, 8, 9, 8, 10, 6, 7]
```
### Syntactic Errors
`nonword_frequency()`
```doctest
import spacy
from syntactic_errors import nonword_frequency

nonword_frequency(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/contains_nonwords.txt", dataset_fp="words_alpha.txt", amount=100)
```
```doctest
1.0186757215619695
```
`avg_num_nonwords()`
```doctest
import spacy
from syntactic_errors import avg_num_nonwords

avg_num_nonwords(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/contains_nonwords.txt", amount=100)
```
```doctest
0.5093378607809848
```
`incorrectly_followed_articles()`
```doctest
import spacy
from syntactic_errors import incorrectly_followed_articles

incorrectly_followed_articles(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
1
```
#### `count_num_sentences_without_verbs()`
```doctest
import spacy
from syntactic_errors import count_num_sentences_without_verbs

count_num_sentences_without_verbs(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
6
```
### Lexical Repetition
#### `most_frequent_word()`
```doctest
import spacy
from lexical_repetition import most_frequent_word

most_frequent_word(nlp=spacy.load('en_ccore_web_lg'), file_path="sample_text/test.txt")
```
```doctest
('life', 5)
```
#### `repeating_unique_word_ratio()`
```doctest
import spacy
from lexical_repetition import repeating_unique_word_ratio

repeating_unique_word_ratio(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
0.9533333333333334
```
#### `total_consecutive_words()`
```doctest
import spacy
from lexical_repetition import total_consecutive_words

total_consecutive_words(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
0
```
### Lexical Variation
#### `windowed_text_token_ratio`
```doctest
import spacy
from lexical_variation import windowed_text_token_ratio

windowed_text_token_ratio(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", window_size=20)
```
```doctest
0.9238095238095237
```
#### `number_of_unique_token()`
```doctest
import spacy
from lexical_variation import number_of_unique_tokens

number_of_unique_tokens(nlp=spacy.load(en_core_web_lg), file_path="sample_text/test.txt")
```
```doctest
311
```
#### `number_of_uniue_lemmas()`
```doctest
import spacy
from lexical_variation import number_of_unique_lemmas

number_of_unique_lemmas(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt")
```
```doctest
278
```
### Similarity
#### `stats_similarity_of_words()`
```doctest
import spacy
from similarity  import stats_similarity_of_words

stats_similarity_of_words(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", window_size=3)
```
```doctest
{'mean': 0.4197374429223744, 'max': 0.7733333333333334, 'min': 0.013333333333333336, 'std': 0.1215306716400124}
```
#### `mean_similarity_of_sentences()`
```doctest
import spacy
from similarity import mean_similarity_of_sentences

mean_similarity_of_sentences(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/sample.txt")
```
```doctest
0.8783600713012477
```
### Term Frequency-Inverse Document Frequency
```doctest
import spacy
from term_freq_inverse_doc_freq import tf_idf

tf_idf(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/test.txt", document_list=["sample_text/sample.txt", "sample_text/test.txt", "sample_text/contains_nonwords.txt"], term="life")
```
```doctest
0.0
```
## Sample Input Files
The folder sample_text contains text documents for convenient testing of these NLP functions.

| File name             | Description                                                                                         | Use                                                                                                                                             |
|-----------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| test.txt              | This is a text file containing an AI generated story of 500 words.                                  | Can be used to test any of the nlp functions.                                                                                                   |
| contains_nonwords.txt | This is a modified version of test.txt, in which nonsensical words were added to several sentences. | This can be used to test functions detecting presence of non-words.                                                                             |
| sample.txt            | This is a subset of test.txt containing the first three sentences of the document.                  | This file is used in most of the Jupyter Notebook examples for quick testing and easy visualization. It is also used to validate some functions. |

## Acknowledgement
- [spaCy](https://spacy.io/): Free and open source library for industrial-strength Natural Language Processing (NLP) in Python