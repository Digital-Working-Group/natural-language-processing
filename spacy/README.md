# Linguistic Features: SpaCy
This repository contains examples of how to use the [spaCy Python library](https://spacy.io/usage/linguistic-features) to generate linguistic features from raw text input.

## Installation and Setup

### Python Requirements

```

```

Check your Python version:
```sh
python --version
```
See [Anaconda](https://www.anaconda.com/download/success) as an option to switch between Python versions. This repository has been tested with Python 3.10.11.

Install requirements for Python 3.10.11:
```sh
pip install -r requirements/py3-10-11/requirements.txt ## Python 3.10.11 requirements
```

Note: you may use the pip install command described above even if you are working with a different Python version, but you may need to adjust the requirements.txt file to fit any dependencies specific to that Python version.

### Requirements.txt License Information
License information for each set of requirements.txt can be found in their respective `pip-licenses.md` file within the requirements/python[version] folders.

### Docker Support
[Docker](https://docs.docker.com/engine/install/) support can be found via the `Dockerfile` and `build_docker.sh` and `run_docker.sh` files.

Please see Docker's documentation for more information ([docker build](https://docs.docker.com/build/), [Dockerfile](https://docs.docker.com/build/concepts/dockerfile/), [docker run](https://docs.docker.com/reference/cli/docker/container/run/)).

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
|-- repeated-words-paragraph.txt ## A single paragraph with repeated words
|-- sentence.txt ## A single sentence
|-- story.txt ## A multi-paragraph story
|-- story_with_nonwords.txt ## A multi-paragraph story with nonwords
```

## Sample Output Files

For each sample text file, each feature will have a separate JSON output file (see [sample_text/pos_tag_ratio/en_core_web_lg](sample_text/pos_tag_ratio/en_core_web_lg)). Additionally, for each sample text file, a single JSON is created with all features (see [sample_text/data/en_core_web_lg](sample_text/data/en_core_web_lg])).

## Extracting Linguistic Features

See [main.py](main.py) for usage examples across all functions.

## Common Inputs

Each function takes in a spaCy model and filepath, which are loaded into a NLPUtil object, see [nlp_utility.py.](nlp_utility.py)

| Parameter | Description | Example |
| - | - | - |
| model | The spaCy model to load and use for tagging parts of speech. | 'en_core_web_lg' |
| filepath | The filepath to a text file to process. | 'sample_text/story.txt' |

## Feature Documentation

### Parts of Speech Tagging

`pos_tagging.pos_tag_ratio()` calculates the tag ratio for every `amount` words. 
```
tag_ratio = tag_ct / total_tokens * amount
```
For example, 8.547 = 60 / 702 * 100, where we have 60 occurrences of a tag (e.g., PRP), 702 total tokens, and have amount is set to 100. The intent is to control for the number of total words/tokens. The amount being set to 100 is arbitrary, if set to 1 then it becomes a decimal percentage.

#### Input

| Parameter | Description | Example |
| - | - | - |
| tag_list | A list of tags to generate the tag ratio for (see `pos_tagging.data_to_df()` for a full list) | ['POS', 'TAG'] |
| amount | The tag ratio is the number of tags divided by the total tokens multiplied by the amount. | 100 |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
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

### Parts of Speech Ratios (Alphanumeric characters only)

`pos_tagging.alpha_pos_ratio()` calculates the ratio of specific part(s) of speech to total words, examining alphanumeric (is_alpha) characters only. Information about the parts-of-speech tags can be found in [spacy_pos_tags_explained.md.](spacy_pos_tags_explained.md) An output JSON is written for every filepath and for every key, value pair in pos_to_list.

#### Input

| Parameter | Description | Example |
| - | - | - |
| pos_to_list | A part of speech string name to the list of associated POS tags. | {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'], 'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']} |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
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

### Parts of Speech Ratios (Alphanumeric characters only): Sentences

`pos_tagging.alpha_pos_ratio_sentences()` iterates over every sentence and calculates the ratio of specific part(s) of speech to total words, examining alphanumeric (is_alpha) characters only. Summary statistics of the ratios across all the sentences are returned. Information about the parts-of-speech tags can be found in [spacy_pos_tags_explained.md.](spacy_pos_tags_explained.md) An output JSON is written for every filepath and for every key, value pair in pos_to_list.

#### Input

| Parameter | Description | Example |
| - | - | - |
| pos_to_list | A part of speech string name to the list of associated POS tags. | {'nouns': ['NOUN', 'PROPN'], 'pronouns': ['PRON'], 'conjunctions': ['CONJ', 'CCONJ', 'SCONJ']} |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
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

Only takes in the [Common Inputs](#common-inputs), which are model and filepath.

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
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

### Generate Noun Features

`semantic_complexity.generate_noun_feature()` serves as a general function for several noun-based features. Each specific noun-based feature utilizes generate_noun_feature() with a different set of keyword arguments.

For example, for each noun, `semantic_complexity.abstractness()` searches [datasets/dataset_for_abstractness.xlsx](datasets/dataset_for_abstractness.xlsx) for the noun (in the 'Word' column) and upon finding it, extracts the 'Conc.M' column's value as the feature (abstractness). If we have a noun ('aardvark') and find it in the 'Word' column, the value in the 'Conc.M' column for 'aardvark' represents the abstractness.

If the noun isn't found in the `word_column`, then its lemma is also searched for in the `word_column`. If neither the noun nor its lemma are found, then it's skipped. The noun (or lemma) is converted into a value based on the `increment_result` function.

#### Input

| Parameter | Description | Example |
| - | - | - |
| feature | The name of the feature. | 'abstractness' |
| dataset_fp | The filepath to the dataset excel sheet needed for the feature calculation. | 'datasets/dataset_for_abstractness.xlsx' |
| read_excel_kwargs | Keyword arguments to add to the excel reading function call (pd.read_excel()) | {'skiprows': 1} |
| word_column | The column in the excel sheet that is searched for an input noun. | 'Word' |
| feature_column | The column in the excel sheet that contains the feature's value. | 'Conc.M' |
| increment_result | A function to retrieve the value from the feature_column. | lambda r: r.item() |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
| data.total_nouns | The total number of nouns. | 10 |
| data.feature_data[N].token.text | The text of the token. | 'things' |
| data.feature_data[N].word | The lowercased version of token.text. The lemma is the base form of a word. If the word (token.text) can't be found in the dataset and its lemma can be found, this will be equal to its lemma. If neither can be found, the word gets skipped and isn't included in the output data. | 'thing' |
| data.feature_data[N].feature_val | The value of the feature found in the feature column. | 0.315 |
| data.feature_data[N].is_lemma | This is equal to 0 if data[N].word is found directly in the word column and is equal to 1 if only the lemma (base form) is found in the word column instead. | 1 |

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

#### Word Prevalence

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

### Tense Inflected Verbs

`syntactic_complexity.tense_inflected_verbs()` generates measures related to the number of tense-inflected verbs and only iterates over alphanumeric tokens. Tense-inflected verbs include present and past verbs, as well as modal auxiliary verbs.

#### Input

| Parameter | Description | Example |
| - | - | - |
| amount | The tiv_ratio is the number of tense-inflected verbs divided by the total tokens multiplied by the amount. | 100 |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
| parameters.function | The name of the function. | See below. |
| data.tiv_ratio | The tiv_ratio is the number of tense-inflected verbs divided by the total tokens multiplied by the amount. When set to 1, it's a decimal percentage and when set to 100, it's a whole percentage. | 15.0 |
| data.total_tivs | The total number of tense-inflected verbs. | 3 |
| data.total_alpha_tokens | The total number of alphanumeric tokens. | 20 |
| data.present_verbs | The total number of present tense verbs. | 3 |
| data.past_verbs | The total number of past tense verbs. | 0 |
| data.modal_auxiliaries | The total number of modal auxiliaries. | 0 |
| data.feature_data[N].token.text | The token's text. | 'know' |
| data.feature_data[N].token.pos_ | The simple part of speech tag. | 'VERB' |
| data.feature_data[N].tense | The list of tense(s) of the token. | ['Pres'] |
| data.feature_data[N].tag | The detailed part of speech tag. | 'VBP' |
| data.feature_data[N].word_type | The word type (present_verb, past_verb, modal_auxiliary) | 'present_verb' |

Please see [sample_text/tense_inflected_verbs/en_core_web_lg](sample_text/tense_inflected_verbs/en_core_web_lg) for sample output.

Here is an excerpt from [sentence.json](sample_text/tense_inflected_verbs/en_core_web_lg/sentence.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/sentence.txt",
        "amount": 100,
        "function": "tense_inflected_verbs"
    },
    "data": {
        "tiv_ratio": 15.0,
        "total_tivs": 3,
        "total_alpha_tokens": 20,
        "present_verbs": 3,
        "past_verbs": 0,
        "modal_auxiliaries": 0,
        "feature_data": [
            {
                "token.text": "know",
                "token.pos_": "VERB",
                "tense": [
                    "Pres"
                ],
                "tag": "VBP",
                "word_type": "present_verb"
            },
            {
                "token.text": "think",
                "token.pos_": "VERB",
                "tense": [
                    "Pres"
                ],
                "tag": "VBP",
                "word_type": "present_verb"
            },
            {
                "token.text": "become",
                "token.pos_": "VERB",
                "tense": [
                    "Pres"
                ],
                "tag": "VBP",
                "word_type": "present_verb"
            }
        ]
    }
}
```

### Dependency Distance

`syntactic_complexity.dependency_distance()` generates dependency distance metrics, which can measure syntactic complexity. The greater the dependency distance is, the more complex it is. The [TextDescriptives](https://github.com/HLasse/TextDescriptives) implementation is utilized here and their specific dependency distance documentation can be found [here](https://hlasse.github.io/TextDescriptives/dependencydistance.html). The TextDescriptives implementation follows the description in ([Oya 2011](https://www.paaljapan.org/conference2011/ProcNewest2011/pdf/poster/P-13.pdf)).

#### Input

An important note is that textdescriptives must be imported to allow for spaCy to access it as a pipe when running this function.

| Parameter | Description | Example |
| - | - | - |
| pipe_list | The list of the pipe(s) needed to access the dependency distance implementation. | ['textdescriptives/dependency_distance'] |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
| parameters.function | The name of the function. | See below. |
| data.dependency_distance_mean | Mean dependency distance on the sentence level. | 2.78 |
| data.dependency_distance_std | Standard deviation of dependency distance on the sentence level. | 0.59 |
| data.prop_adjacent_dependency_relation_mean | Mean proportion of adjacent dependency relations on the sentence level. | 0.44 |
| data.prop_adjacent_dependency_relation_std | Standard deviation of the proportion of adjacent dependency relations on the sentence level. | 0.09 |

Please see [sample_text/dependency_distances/en_core_web_lg](sample_text/dependency_distance/en_core_web_lg) for sample output.

Here is an excerpt from [story.json](sample_text/dependency_distance/en_core_web_lg/story.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/story.txt",
        "function": "dependency_distance"
    },
    "data": {
        "dependency_distance_mean": 2.7765067920765767,
        "dependency_distance_std": 0.5908581635641064,
        "prop_adjacent_dependency_relation_mean": 0.43699129406789733,
        "prop_adjacent_dependency_relation_std": 0.08557102852242565
    }
}
```

### Moving Type Token Ratio

`lexical_variation.moving_type_token_ratio()` calculates the type/token ratio for a fixed-length window, moving one word at a time and returns data per window and overall statistics. It iterates only over alphanumeric tokens. It will raise an AssertionError if the total number of alphanumeric tokens are less than the window_size. This implementation attempts to follow the description in section 2.4 (Lexical Measures) of [(Cho 2022, Automated analysis of lexical features in Frontotemporal Degeneration)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8044033/)

#### Input

| Parameter | Description | Example |
| - | - | - |
| window_size | The size of the moving window. | 20 |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
| parameters.function | The name of the function. | See below. |
| data.average_type_token_ratio | The average of the type token ratio across all windows. | 0.97 |
| data.num_windows | The total number of windows. | 37 |
| data.windows[N].num_unique_words | The total unique words in the window. | 19 |
| data.windows[N].type_token_ratio | The num_unique_words / the number of words in the window (window_size) | 0.95 |

Please see [sample_text/moving_type_token_ratio/en_core_web_lg](sample_text/moving_type_token_ratio/en_core_web_lg) for sample output.

Here is an excerpt from [paragraph.json](sample_text/moving_type_token_ratio/en_core_web_lg/paragraph.json):

```yaml
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/paragraph.txt",
        "window_size": 20,
        "function": "moving_type_token_ratio"
    },
    "data": {
        "average_type_token_ratio": 0.9689189189189189,
        "num_windows": 37,
        "windows": [
            {
                "num_unique_words": 19,
                "type_token_ratio": 0.95
            },
            {
                "num_unique_words": 19,
                "type_token_ratio": 0.95
            },
            {
                "num_unique_words": 19,
                "type_token_ratio": 0.95
            },
        ...
            {
                "num_unique_words": 19,
                "type_token_ratio": 0.95
            }
        ]
    }
}
```

### Nonword Frequency

`syntactic_errors.nonword_frequency()` calculates the frequency of non-words per amount of words. Non-words are defined as the token's text or lemma not being in the dataset. The dataset ([datasets/words_alpha.txt](datasets/words_alpha.txt)) comes from [Kaggle](https://www.kaggle.com/datasets/bwandowando/479k-english-words)) and contains over 466K English words.

#### Input

| Parameter | Description | Example |
| - | - | - |
| amount | The nonword frequency is the total number of non-words divided by the total words multiplied by amount. | 100 |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
| parameters.function | The name of the function. | See below. |

Please see [sample_text/nonword_frequency/en_core_web_lg](sample_text/nonword_frequency/en_core_web_lg) for sample output.

Here is an excerpt from [story.json](sample_text/nonword_frequency/en_core_web_lg/story.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/story.txt",
        "dataset_fp": "datasets/words_alpha.txt",
        "amount": 100,
        "function": "nonword_frequency"
    },
    "data": {
        "total_nonwords": 3,
        "total_words": 586,
        "nonword_frequency": 0.5119453924914675
    }
}
```

### Word Repetition

`lexical_repetition.word_repetition()` iterates over alphanumeric tokens and calculates several word-based repetition metrics.

#### Input

Only takes in the [Common Inputs](#common-inputs), which are model and filepath.

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
| parameters.function | The name of the function. | See below. |
| data.most_frequent_word | The most frequently occurring word. | i |
| data.most_frequent_ct | The number of times that the most frequent word occurs. | 34 |
| data.repeating_words | The total number of words that appear >1 times. | 393 |
| data.unique_words | The total number of unique words. | 288 |
| data.repeating_to_unique_ratio | repeating_words / unique_words | 1.36 |
| data.consecutive_words | The total number of words that appear consecutively. | 0 |

Please see [sample_text/word_repetition/en_core_web_lg](sample_text/word_repetition/en_core_web_lg) for sample output.

Here is an excerpt from [repeated-words-paragraph.json](sample_text/word_repetition/en_core_web_lg/repeated-words-paragraph.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/repeated-words-paragraph.txt",
        "function": "word_repetition"
    },
    "data": {
        "most_frequent_word": "the",
        "most_frequent_ct": 6,
        "repeating_words": 33,
        "unique_words": 54,
        "repeating_to_unique_ratio": 0.6111111111111112,
        "consecutive_words": 5
    }
}
```

### Doc Similarity

`similarity.doc_similarity()` calculates word similarity across a sliding window over a document's alphanumeric tokens.

#### Input

| Parameter | Description | Example |
| - | - | - |
| window_size | The size of the sliding window. | 3 |

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
| parameters.function | The name of the function. | See below. |
| data.avg_similarity_score | The average of all the windows' similarity scores. | 0.526 |
| data.max_similarity_score | The maximum of all the windows' similarity scores. | 0.723 |
| data.min_similarity_score | The minimum of all the windows' similarity scores. | 0.393 |
| data.std_similarity_score | The standard deviation of all the windows' similarity scores. | 0.09 |
| data.num_windows | The number of windows. | 18 |
| data.windows[N].avg_similarity_score | The average similarity score of the window. | 0.723 |
| data.windows[N].max_similarity_score | The maximum similarity score of the window. | 0.81 |
| data.windows[N].min_similarity_score | The minimum similarity score of the window. | 0.68 |
| data.windows[N].std_similarity_score | The standard deviation of similarity scores of the window. | 0.07 |

Please see [sample_text/doc_similarity/en_core_web_lg](sample_text/doc_similarity/en_core_web_lg) for sample output.

Here is an excerpt from [sentence.json](sample_text/doc_similarity/en_core_web_lg/sentence.json):

```yaml
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/sentence.txt",
        "window_size": 3,
        "function": "doc_similarity"
    },
    "data": {
        "avg_similarity_score": 0.5261111111111112,
        "max_similarity_score": 0.7233333333333335,
        "min_similarity_score": 0.39333333333333337,
        "std_similarity_score": 0.09275204136012637,
        "num_windows": 18,
        "windows": [
            {
                "avg_similarity_score": 0.7233333333333335,
                "max_similarity_score": 0.81,
                "min_similarity_score": 0.68,
                "std_similarity_score": 0.07505553499465135
            },
        ....
            {
                "avg_similarity_score": 0.5266666666666667,
                "max_similarity_score": 0.54,
                "min_similarity_score": 0.52,
                "std_similarity_score": 0.011547005383792526
            }
        ]
    }
}
```

### Sent Similarity

`similarity.sent_similarity()` calculates word similarity of sentences over a document's alphanumeric tokens.

#### Input

Only takes in the [Common Inputs](#common-inputs), which are model and filepath.

#### Output

| Key | Description | Example |
| - | - | - |
| parameters | The parameters of the function. | See the Input table and below. |
| parameters.function | The name of the function. | See below. |
| data.avg_similarity_score | The average of all the sentences' similarity scores. | 0.859 |
| data.max_similarity_score | The maximum of all the sentences' similarity scores. | 0.97 |
| data.min_similarity_score | The minimum of all the sentences' similarity scores. | 0.66 |
| data.std_similarity_score | The standard deviation of all the sentences' similarity scores. | 0.06 |
| data.num_sentences | The total number of sentences in the document. | 34 |

Please see [sample_text/sent_similarity/en_core_web_lg](sample_text/sent_similarity/en_core_web_lg) for sample output.

Here is an excerpt from [story.json](sample_text/sent_similarity/en_core_web_lg/story.json):

```json
{
    "parameters": {
        "model": "en_core_web_lg",
        "filepath": "sample_text/story.txt",
        "function": "sent_similarity"
    },
    "data": {
        "avg_similarity_score": 0.8591443850267388,
        "max_similarity_score": 0.97,
        "min_similarity_score": 0.66,
        "std_similarity_score": 0.0586939596675117,
        "num_sentences": 34
    }
}
```

## Acknowledgement
- [spaCy](https://spacy.io/): Free and open source library for industrial-strength Natural Language Processing (NLP) in Python