# Linguistic Features: SpaCy
This repository contains examples of how to use the [spaCy Python library](https://spacy.io/usage/linguistic-features) to generate linguistic features from raw text input.

## Installation
### Without Docker
Check your python version:
```
python --version
```
Install requirements for python 3.13.5:
```
pip install -r python3-15-5_requirements.txt
```
## Data Dictionary
See [data_dictionary.md](data_dictionary.md) for detailed descriptions of each linguistic feature, the function(s) used to generate them, and references to the papers they were extracted from.

## Jupyter Notebook Examples
Please run [jupyter notebook](https://docs.jupyter.org/en/latest/running.html) and see [nlp_function_examples.ipynb](nlp_function_examples.ipynb) for an interactive set of examples. Also, see the usage example sections below.

## Load spaCy models
Prior to being able to load spaCy models such as [en_core_web_lg](https://spacy.io/models/en#en_core_web_lg), one must run `spacy download <model name`:

```sh
spacy download en_core_web_lg
```

Downloading the model allows for loading via spaCy:
```python
nlp = spacy.load('en_core_web_lg')
```

Without downloading the model, one may encounter errors such as:
```
OSError: [E050] Can't find model 'en_core_web_lg'. It doesn't seem to be a Python package or a valid path to a data directory.
```
## Sample Input Text Files
```
sample_text
|-- paragraph.txt ## A single paragraph
|-- sentence.txt ## A single sentence
|-- story.txt ## A multi-paragraph story
|-- story_with_nonwords.txt ## A multi-paragraph story with nonwords
```

## Extracting Linguistic Features

See [main.py](main.py) for usage examples.

### Parts of Speech Tagging

`pos_tagging.pos_tag_ratio()` calculates the tag ratio for every `amount` words. 
```
tag_ratio = tag_ct / total_tokens * amount
```
For example, 8.547 = 60 / 702 * 100, where we have 60 occurrences of a tag (e.g., PRP), 702 total tokens, and have amount is set to 100. The intent is to control for the number of total words/tokens. The amount being set to 100 is arbitrary, if set to 1 then it becomes a normal percentage.

#### Input

| Parameter | Description | Example |
| - | - | - |
| model | The spaCy model to load and use for tagging parts of speech. | 'en_core_web_lg' |
| filepath | The filepath to a text file to process. | 'sample_text/story.txt' |
| tag_list | A list of tags to generate the tag ratio for (see `pos_tagging.data_to_df()` for a full list) | ['POS', 'TAG'] |
| amount | The tag ratio is the number of tags divided by the total tokens multipled by the amount. | 100 |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The list of parameters to the function. | See Input table and below. |
| parameters.function | The name of the function. | See below. |
| data[N].tag | The tag used as an input to pos_tag_ratio(), an element of tag_list. | "POS" |
| data[N].tag_data.tag_label | The tag's label. | "PRON" |
| data[N].tag_data.tag_ratio | The tag's ratio per amount of words. | 13.675 |
| data[n].tag_data.spacy.explain | The output of spacy.explain(tag_label). | "pronoun" |

Please see [sample_text/pos_tag_ratio/en_core_web_lg](sample_text/pos_tag_ratio/en_core_web_lg) for sample output.

Here is an excerpt from [story.json](sample_text/pos_tag_ratio/en_core_web_lg/story.json):

```yaml
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/story.txt",
        "tag_list": [
            "POS",
            "TAG"
        ],
        "amount": 100,
        "function": "pos_tag_ratio"
    },
    "data": [
        {
            "tag": "POS",
            "total_tokens": 702,
            "tag_data": [
                {
                    "tag_label": "PRON",
                    "tag_ct": 96,
                    "tag_ratio": 13.675213675213676,
                    "spacy.explain": "pronoun"
                },
                {
                    "tag_label": "VERB",
                    "tag_ct": 98,
                    "tag_ratio": 13.96011396011396,
                    "spacy.explain": "verb"
                },
        ...
            "tag": "TAG",
            "total_tokens": 702,
            "tag_data": [
                {
                    "tag_label": "PRP",
                    "tag_ct": 60,
                    "tag_ratio": 8.547008547008547,
                    "spacy.explain": "pronoun, personal"
                },
                {
                    "tag_label": "VBP",
                    "tag_ct": 12,
                    "tag_ratio": 1.7094017094017095,
                    "spacy.explain": "verb, non-3rd person singular present"
                },
        ...

}
```

#### Sample Usage

```py
import pos_tagging as pos_t

def pos_tag_ratio():
    """
    run pos_tagging.pos_tag_ratio()
    """
    model = 'en_core_web_lg'
    tag_list = ['POS', 'TAG']
    amount = 100
    sentence = 'sample_text/sentence.txt'
    paragraph = 'sample_text/paragraph.txt'
    story = 'sample_text/story.txt'
    sample_files = [sentence, paragraph, story]
    for filepath in sample_files:
        pos_t.pos_tag_ratio(model, filepath, tag_list, amount=amount)
```

### Parts of Speech Ratios (Alphanumeric characters only)

`pos_tagging.alpha_pos_ratio()` calculates the ratio of specific part(s) of speech to total words, examining alphanumeric (is_alpha) characters only. Information about the parts-of-speech tags can be found in [spacy_pos_tags_explained.md.](spacy_pos_tags_explained.md) An output JSON is written for every filepath and for every key, value pair in pos_to_list.

#### Input

| Parameter | Description | Example |
| - | - | - |
| model | The spaCy model to load and use for tagging parts of speech. | 'en_core_web_lg' |
| filepath | The filepath to a text file to process. | 'sample_text/story.txt' |
| pos_to_list | A part of speech string name to the list of associated POS tags. | {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'], 'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']} |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The list of parameters to the function. | See Input table and below. |
| parameters.function | The name of the function. | See below. |
| data.pos_ratio | Contains the POS ratio across the whole document. | 0.229 |

Please see [sample_text/alpha_pos_ratio/en_core_web_lg](sample_text/alpha_pos_ratio/en_core_web_lg) for sample output.

Here is an excerpt from [story_nouns.json](sample_text/alpha_pos_ratio/en_core_web_lg/story_nouns.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/story.txt",
        "pos_list": [
            "NOUN",
            "PROPN"
        ],
        "function": "alpha_pos_ratio"
    },
    "data": {
        "pos_ratio": 0.22866894197952217
    }
}
```

#### Sample Usage

```py
import pos_tagging as pos_t

def alpha_pos_ratio():
    """
    run pos_tagging.alpha_pos_ratio()
    """
    model = 'en_core_web_lg'
    sentence = 'sample_text/sentence.txt'
    paragraph = 'sample_text/paragraph.txt'
    story = 'sample_text/story.txt'
    sample_files = [sentence, paragraph, story]
    pos_to_list = {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'],
        'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}
    for filepath in sample_files:
        pos_t.alpha_pos_ratio(model, filepath, pos_to_list=pos_to_list)
```

### Parts of Speech Ratios (Alphanumeric characters only): Sentences

`pos_tagging.alpha_pos_ratio_sentences()` iterates over every sentence and calculates the ratio of specific part(s) of speech to total words, examining alphanumeric (is_alpha) characters only. Summary statistics of the ratios across all the sentences are returned. Information about the parts-of-speech tags can be found in [spacy_pos_tags_explained.md.](spacy_pos_tags_explained.md) An output JSON is written for every filepath and for every key, value pair in pos_to_list.

#### Input

| Parameter | Description | Example |
| - | - | - |
| model | The spaCy model to load and use for tagging parts of speech. | 'en_core_web_lg' |
| filepath | The filepath to a text file to process. | 'sample_text/story.txt' |
| pos_to_list | A part of speech string name to the list of associated POS tags. | {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'], 'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']} |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The list of parameters to the function. | See Input table and below. |
| parameters.function | The name of the function. | See below. |
| data.sent_mean | Mean of POS ratio across all sentences. | 0.246 |
| data.sent_max | Max of POS ratio across all sentences. | 0.4 |
| data.sent_min | Min of POS ratio across all sentences. | 0.09 |
| data.sent_std | Standard deviation of POS ratio across all sentences. If there are <2 sentences, is set to None. | 0.08 |
| data.sent_total | Total number of sentences. | 33 |

Please see [sample_text/alpha_pos_ratio/en_core_web_lg](sample_text/alpha_pos_ratio_sentences/en_core_web_lg) for sample output.

Here is an excerpt from [story_nouns.json](sample_text/alpha_pos_ratio_sentences/en_core_web_lg/story_nouns.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/story.txt",
        "pos_list": [
            "NOUN",
            "PROPN"
        ],
        "function": "alpha_pos_ratio_sentences"
    },
    "data": {
        "sent_mean": 0.2456333989784595,
        "sent_max": 0.4,
        "sent_min": 0.09090909090909091,
        "sent_std": 0.08429027819940069,
        "sent_total": 33
    }
}
```

#### Sample Usage

```py
import pos_tagging as pos_t

def alpha_pos_ratio_sentences():
    """
    run pos_tagging.alpha_pos_ratio_sentences()
    """
    model = 'en_core_web_lg'
    sentence = 'sample_text/sentence.txt'
    paragraph = 'sample_text/paragraph.txt'
    story = 'sample_text/story.txt'
    sample_files = [sentence, paragraph, story]
    pos_to_list = {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'],
        'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']}
    for filepath in sample_files:
        pos_t.alpha_pos_ratio_sentences(model, filepath, pos_to_list=pos_to_list)
```

### Idea Density: Sentences

`semantic_complexity.idea_density_sentences()` iterates over every sentence and calculates idea density, examining alphanumeric (is_alpha) characters only. Idea density is the number of propositions divided by the total words in the sentence. Idea density is also known as propositional density (or P-density).

Propositions include verbs, adjectives, adverbs, prepositions, and conjunctions.
- VERB: verbs
- ADJ: adjectives
- ADV: adverbs
- ADP: adposition (prepositions and postpositions)
- CONJ: conjunction
- CCONJ: coordinating conjunction
- SCONJ: subordinating conjunction

Information about the parts-of-speech tags can be found in [spacy_pos_tags_explained.md.](spacy_pos_tags_explained.md)

#### Input

| Parameter | Description | Example |
| - | - | - |
| model | The spaCy model to load and use for tagging parts of speech. | 'en_core_web_lg' |
| filepath | The filepath to a text file to process. | 'sample_text/story.txt' |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The list of parameters to the function. | See Input table and below. |
| parameters.function | The name of the function. | See below. |
| data.total_sentences | The total number of sentences in the text. | 3 |
| data.sent_list[N].sent_idx | The sentence index number (zero is the first sentence). | 0 |
| data.sent_list[N].total_tokens | The total number of tokens in the sentence. | 14 |
| data.sent_list[N].idea_density | The idea density of the sentence. |

Please see [sample_text/idea_density_sentences/en_core_web_lg](sample_text/idea_density_sentences/en_core_web_lg) for sample output.

Here is an excerpt from [paragraph.json](sample_text/idea_density_sentences/en_core_web_lg/paragraph.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/paragraph.txt",
        "function": "idea_density_sentences"
    },
    "data": {
        "total_sentences": 3,
        "sent_list": [
            {
                "sent_idx": 0,
                "total_tokens": 20,
                "idea_density": 0.55
            },
            {
                "sent_idx": 1,
                "total_tokens": 14,
                "idea_density": 0.5714285714285714
            },
            {
                "sent_idx": 2,
                "total_tokens": 22,
                "idea_density": 0.4090909090909091
            }
        ]
    }
}
```

#### Sample Usage

```py
import semantic_complexity as sem_c

def idea_density_sentences():
    """
    run semantic_complexity.idea_density_sentences()
    """
    model = 'en_core_web_lg'
    sentence = 'sample_text/sentence.txt'
    paragraph = 'sample_text/paragraph.txt'
    story = 'sample_text/story.txt'
    sample_files = [sentence, paragraph, story]
    for filepath in sample_files:
        sem_c.idea_density_sentences(model, filepath)
```

### Generate Noun Features

`semantic_complexity.generate_noun_feature()` serves as a general function for several noun-based features. Each specific noun-based feature utilizes generate_noun_feature() with a different set of keyword arguments.

For example, for each noun, `semantic_complexity.abstractness()` searches [datasets/dataset_for_abstractness.xlsx](datasets/dataset_for_abstractness.xlsx) for the noun (in the 'Word' column) and upon finding it, extracts the 'Conc.M' column's value as the feature (abstractness). If we have a noun ('aardvark') and find it in the 'Word' column, the value in the 'Conc.M' column for 'aardvark' represents the abstractness.

If the noun isn't found in the `word_column`, then its lemma is also searched for in the `word_column`. If neither the noun nor its lemma are found, then it's skipped. The noun (or lemma) is converted into a value based on the `increment_result` function.

#### Input

| Parameter | Description | Example |
| - | - | - |
| model | The spaCy model to load and use for tagging parts of speech. | 'en_core_web_lg' |
| filepath | The filepath to a text file to process. | 'sample_text/story.txt' |
| feature | The name of the feature. | 'abstractness' |
| dataset_fp | The filepath to the dataset excel sheet needed for the feature calculation. | 'datasets/dataset_for_abstractness.xlsx' |
| read_excel_kwargs | Keyword arguments to add to the excel reading function call (pd.read_excel()) | {'skiprows': 1} |
| word_column | The column in the excel sheet that is searched for an input noun. | 'Word' |
| feature_column | The column in the excel sheet that contains the feature's value. | 'Conc.M' |
| increment_result | A function to retrieve the value from the feature_column. | lambda r: r.item() |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The list of parameters to the function. | See Input table and below. |
| parameters.function | The name of the function. | See below. |
| data.total_nouns | The total number of nouns. | 10 |
| data.feature_data[N].token.text | The text of the token. | 'things' |
| data.feature_data[N].word | The lowercased version of token.text. It will be the lemma (base form) of the token.text, if the token.text can't be found in the dataset and only the lemma can be found. | 'thing' |
| data.feature_data[N].feature_val | The value of the feature found in the feature column. | 0.315 |
| data.feature_data[N].is_lemma | This is equal to 0 if data[N].word is found directly in the word column and is equal to 1 if only the lemma (base form) is found in the word column instead. | 1 |

#### Sample Usage

```py
import semantic_complexity as sem_c

def generate_noun_features():
    """
    run the several generate_noun_feature-based functions from semantic_complexity.py
    """
    model = 'en_core_web_lg'
    sample_files = get_sample_files()
    for filepath in sample_files:
        sem_c.abstractness(model, filepath)
        sem_c.semantic_ambiguity(model, filepath)
        sem_c.word_frequency(model, filepath)
        sem_c.word_prevalence(model, filepath)
        sem_c.word_familiarity(model, filepath)
        sem_c.age_of_acquisition(model, filepath)
```

#### Abstractness

The 'Conc.M' column in the dataset represents the concreteness of word. Here, we take the inverse (1/concreteness) for each noun.

Please see [sample_text/abstractness/en_core_web_lg](sample_text/abstractness/en_core_web_lg) for sample output.

Here is an excerpt from [paragraph.json](sample_text/abstractness/en_core_web_lg/paragraph.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/paragraph.txt",
        "feature_column": "Conc.M",
        "dataset_fp": "datasets/dataset_for_abstractness.xlsx",
        "word_column": "Word",
        "feature": "abstractness"
    },
    "data": {
        "total_nouns": 10,
        "feature_data": [
            {
                "token.text": "life",
                "word": "life",
                "feature_val": 0.3717472118959108,
                "is_lemma": 0
            },
            {
                "token.text": "things",
                "word": "thing",
                "feature_val": 0.31545741324921134,
                "is_lemma": 1
            },
            {
                "token.text": "town",
                "word": "town",
                "feature_val": 0.21551724137931036,
                "is_lemma": 0
            },
            {
                "token.text": "countryside",
                "word": "countryside",
                "feature_val": 0.2232142857142857,
                "is_lemma": 0
            },
            {
                "token.text": "hills",
                "word": "hill",
                "feature_val": 0.20283975659229211,
                "is_lemma": 1
            },
            {
                "token.text": "couple",
                "word": "couple",
                "feature_val": 0.2544529262086514,
                "is_lemma": 0
            },
            {
                "token.text": "farms",
                "word": "farm",
                "feature_val": 0.2178649237472767,
                "is_lemma": 1
            },
            {
                "token.text": "street",
                "word": "street",
                "feature_val": 0.21052631578947367,
                "is_lemma": 0
            },
            {
                "token.text": "diner",
                "word": "diner",
                "feature_val": 0.20746887966804978,
                "is_lemma": 0
            },
            {
                "token.text": "name",
                "word": "name",
                "feature_val": 0.2857142857142857,
                "is_lemma": 0
            }
        ]
    }
}
```

#### Age of Acquisition

The 'AoA' column in the dataset represents the age when a given word is typically learned.

Please see [sample_text/age_of_acquisition/en_core_web_lg](sample_text/age_of_acquisition/en_core_web_lg) for sample output.

Here is an excerpt from [paragraph.json](sample_text/age_of_acquisition/en_core_web_lg/paragraph.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/paragraph.txt",
        "feature_column": "AoA",
        "dataset_fp": "datasets/dataset_for_age_of_acquisition.xlsx",
        "word_column": "Word",
        "feature": "age_of_acquisition"
    },
    "data": {
        "total_nouns": 10,
        "feature_data": [
            {
                "token.text": "life",
                "word": "life",
                "feature_val": 5.666149999999999,
                "is_lemma": 0
            },
            {
                "token.text": "things",
                "word": "things",
                "feature_val": 4.317719583333333,
                "is_lemma": 0
            },
            {
                "token.text": "town",
                "word": "town",
                "feature_val": 5.47843125,
                "is_lemma": 0
            },
            {
                "token.text": "countryside",
                "word": "countryside",
                "feature_val": 8.0,
                "is_lemma": 0
            },
            {
                "token.text": "hills",
                "word": "hills",
                "feature_val": 4.490893750000001,
                "is_lemma": 0
            },
            {
                "token.text": "couple",
                "word": "couple",
                "feature_val": 6.133825353535353,
                "is_lemma": 0
            },
            {
                "token.text": "farms",
                "word": "farms",
                "feature_val": 4.51294625,
                "is_lemma": 0
            },
            {
                "token.text": "street",
                "word": "street",
                "feature_val": 4.600627849462366,
                "is_lemma": 0
            },
            {
                "token.text": "diner",
                "word": "diner",
                "feature_val": 7.30250505050505,
                "is_lemma": 0
            },
            {
                "token.text": "name",
                "word": "name",
                "feature_val": 3.7675083333333332,
                "is_lemma": 0
            }
        ]
    }
}
```

#### Semantic Ambiguity

The 'SemD' column in the dataset represents the degree to which the different contexts associated with a given word vary in their meanings.

Please see [sample_text/semantic_ambiguity/en_core_web_lg](sample_text/semantic_ambiguity/en_core_web_lg) for sample output.

Here is an excerpt from [paragraph.json](sample_text/semantic_ambiguity/en_core_web_lg/paragraph.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/paragraph.txt",
        "feature_column": "SemD",
        "dataset_fp": "datasets/dataset_for_semantic_ambiguity.xlsx",
        "read_excel_kwargs": {
            "skiprows": 1
        },
        "word_column": "!term",
        "feature": "semantic_ambiguity"
    },
    "data": {
        "total_nouns": 10,
        "feature_data": [
            {
                "token.text": "life",
                "word": "life",
                "feature_val": 2.127011833336849,
                "is_lemma": 0
            },
            {
                "token.text": "things",
                "word": "things",
                "feature_val": 1.9507460331936715,
                "is_lemma": 0
            },
            {
                "token.text": "town",
                "word": "town",
                "feature_val": 1.850260277139322,
                "is_lemma": 0
            },
            {
                "token.text": "countryside",
                "word": "countryside",
                "feature_val": 1.548715174128924,
                "is_lemma": 0
            },
            {
                "token.text": "hills",
                "word": "hills",
                "feature_val": 1.6131345963861938,
                "is_lemma": 0
            },
            {
                "token.text": "couple",
                "word": "couple",
                "feature_val": 1.8707156729919414,
                "is_lemma": 0
            },
            {
                "token.text": "farms",
                "word": "farms",
                "feature_val": 1.2653946867156831,
                "is_lemma": 0
            },
            {
                "token.text": "street",
                "word": "street",
                "feature_val": 1.7917702201526633,
                "is_lemma": 0
            },
            {
                "token.text": "diner",
                "word": "diner",
                "feature_val": 1.4845310730241417,
                "is_lemma": 0
            },
            {
                "token.text": "name",
                "word": "name",
                "feature_val": 2.0526260736975077,
                "is_lemma": 0
            }
        ]
    }
}
```

#### Word Familiarity

The 'Pknown' column in the dataset represents a z-standardized measure of the number of people who know a given word.

Please see [sample_text/word_familiarity/en_core_web_lg](sample_text/word_familiarity/en_core_web_lg) for sample output.

Here is an excerpt from [paragraph.json](sample_text/word_familiarity/en_core_web_lg/paragraph.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/paragraph.txt",
        "feature_column": "Pknown",
        "dataset_fp": "datasets/dataset_for_word_prevalence_and_familiarity.xlsx",
        "word_column": "Word",
        "feature": "word_familiarity"
    },
    "data": {
        "total_nouns": 10,
        "feature_data": [
            {
                "token.text": "life",
                "word": "life",
                "feature_val": 0.9976744186046509,
                "is_lemma": 0
            },
            {
                "token.text": "things",
                "word": "thing",
                "feature_val": 1.0,
                "is_lemma": 1
            },
            {
                "token.text": "town",
                "word": "town",
                "feature_val": 0.9976190476190478,
                "is_lemma": 0
            },
            {
                "token.text": "countryside",
                "word": "countryside",
                "feature_val": 0.9905882352941178,
                "is_lemma": 0
            },
            {
                "token.text": "hills",
                "word": "hill",
                "feature_val": 0.9954337899543376,
                "is_lemma": 1
            },
            {
                "token.text": "couple",
                "word": "couple",
                "feature_val": 0.9907621247113163,
                "is_lemma": 0
            },
            {
                "token.text": "farms",
                "word": "farm",
                "feature_val": 0.9978494623655912,
                "is_lemma": 1
            },
            {
                "token.text": "street",
                "word": "street",
                "feature_val": 0.9973614775725596,
                "is_lemma": 0
            },
            {
                "token.text": "diner",
                "word": "diner",
                "feature_val": 0.9977011494252872,
                "is_lemma": 0
            },
            {
                "token.text": "name",
                "word": "name",
                "feature_val": 0.993993993993994,
                "is_lemma": 0
            }
        ]
    }
}
```

#### Word Frequency

The 'Lg10WF' column in the dataset represents the frequency of a given word per million words on a log10 scale.

Please see [sample_text/word_frequency/en_core_web_lg](sample_text/word_frequency/en_core_web_lg) for sample output.

Here is an excerpt from [paragraph.json](sample_text/word_frequency/en_core_web_lg/paragraph.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/paragraph.txt",
        "feature_column": "Lg10WF",
        "dataset_fp": "datasets/dataset_for_word_frequency.xlsx",
        "word_column": "Word",
        "feature": "word_frequency"
    },
    "data": {
        "total_nouns": 10,
        "feature_data": [
            {
                "token.text": "life",
                "word": "life",
                "feature_val": 4.608846822326411,
                "is_lemma": 0
            },
            {
                "token.text": "things",
                "word": "things",
                "feature_val": 4.548241966406093,
                "is_lemma": 0
            },
            {
                "token.text": "town",
                "word": "town",
                "feature_val": 4.101918833680424,
                "is_lemma": 0
            },
            {
                "token.text": "countryside",
                "word": "countryside",
                "feature_val": 2.2576785748691846,
                "is_lemma": 0
            },
            {
                "token.text": "hills",
                "word": "hills",
                "feature_val": 3.011993114659257,
                "is_lemma": 0
            },
            {
                "token.text": "couple",
                "word": "couple",
                "feature_val": 4.056714329516394,
                "is_lemma": 0
            },
            {
                "token.text": "farms",
                "word": "farms",
                "feature_val": 2.164352855784437,
                "is_lemma": 0
            },
            {
                "token.text": "street",
                "word": "street",
                "feature_val": 3.878406887580996,
                "is_lemma": 0
            },
            {
                "token.text": "diner",
                "word": "diner",
                "feature_val": 2.801403710017355,
                "is_lemma": 0
            },
            {
                "token.text": "name",
                "word": "name",
                "feature_val": 4.515025612032066,
                "is_lemma": 0
            }
        ]
    }
}
```

#### Word Frequency

The 'Prevalence' column in the dataset represents the number of people who knew a given word. It was crowdsourced via a study involving over 220,000 people. ([Word prevalence norms for 62,000 English lemmas](https://link.springer.com/article/10.3758/s13428-018-1077-9))

Please see [sample_text/word_prevalence/en_core_web_lg](sample_text/word_prevalence/en_core_web_lg) for sample output.

Here is an excerpt from [paragraph.json](sample_text/word_prevalence/en_core_web_lg/paragraph.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/paragraph.txt",
        "feature_column": "Prevalence",
        "dataset_fp": "datasets/dataset_for_word_prevalence_and_familiarity.xlsx",
        "word_column": "Word",
        "feature": "word_prevalence"
    },
    "data": {
        "total_nouns": 10,
        "feature_data": [
            {
                "token.text": "life",
                "word": "life",
                "feature_val": 2.4420369643248447,
                "is_lemma": 0
            },
            {
                "token.text": "things",
                "word": "thing",
                "feature_val": 2.5758293035489,
                "is_lemma": 1
            },
            {
                "token.text": "town",
                "word": "town",
                "feature_val": 2.4393358558940506,
                "is_lemma": 0
            },
            {
                "token.text": "countryside",
                "word": "countryside",
                "feature_val": 2.188471238006659,
                "is_lemma": 0
            },
            {
                "token.text": "hills",
                "word": "hill",
                "feature_val": 2.3447254310991066,
                "is_lemma": 1
            },
            {
                "token.text": "couple",
                "word": "couple",
                "feature_val": 2.193227487163475,
                "is_lemma": 0
            },
            {
                "token.text": "farms",
                "word": "farm",
                "feature_val": 2.4506951640221115,
                "is_lemma": 1
            },
            {
                "token.text": "street",
                "word": "street",
                "feature_val": 2.4269995579074632,
                "is_lemma": 0
            },
            {
                "token.text": "diner",
                "word": "diner",
                "feature_val": 2.4433473510218704,
                "is_lemma": 0
            },
            {
                "token.text": "name",
                "word": "name",
                "feature_val": 2.2922383711418757,
                "is_lemma": 0
            }
        ]
    }
}
```

## Extracting Linguistic Features
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