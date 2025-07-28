import spacy
import pathlib
import pandas as pd


def data_to_df(doc):
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
    return pd.DataFrame(d)


def POS_ratio(doc, pos='POS', amount=100):
    df = data_to_df(doc)
    pos_count_per_amount = {}
    total = 0
    for pos in df[pos]:
        if pos not in pos_count_per_amount:
            pos_count_per_amount[pos] = 1
        else:
            pos_count_per_amount[pos] += 1
        total += 1
    for pos in pos_count_per_amount:
        pos_count_per_amount[pos] = pos_count_per_amount[pos] / total * amount

    return pos_count_per_amount

def num_tense_inflected_verbs(doc, amount=100):
    total_words = 0
    present_verbs = []
    past_verbs = []
    modal_auxiliary = []
    for token in doc:
        if token.is_alpha:
            total_words += 1

            if token.pos_ == "VERB" and token.morph.get("Tense") == ["Pres"]:
                present_verbs.append(token.morph)
            if token.pos_ == "VERB" and token.morph.get("Tense") == ["Past"]:
                past_verbs.append(token.morph)
            if token.pos_ == "AUX" and token.tag_ == "MD":
                modal_auxiliary.append(token.morph)

            num_tiv = len(present_verbs) + len(modal_auxiliary) + len(past_verbs)
    return num_tiv / total_words * amount

def idea_density(doc, amount=1):
    count_props = 0
    idea_density_sentences = 0
    num_sentences = 0
    for sent in doc.sents:
        num_sentences += 1
        count_words = 0
        for token in sent:
            if token.is_alpha:
                count_words += 1
                if token.pos_ == "PROPN":
                    count_props += 1
        idea_density_sentences += count_props / count_words
    idea_density = idea_density_sentences / num_sentences * amount
    return idea_density

def abstractness(doc, amount=100):
    df = pd.read_excel('13428_2013_403_MOESM1_ESM.xlsx')
    df['Word'] = df['Word'].str.lower()

    total_words = 0
    total_concreteness_values = 0

    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['Word'] == word, 'Conc.M']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['Word'] == word_lemma, 'Conc.M']
                if not result.empty:
                    total_concreteness_values += result.item()
                elif not result_lemma.empty:
                    total_concreteness_values += result_lemma.item()
                else:
                    total_words -= 1
    return total_concreteness_values / total_words * amount

def semantic_ambiguity(doc, amount=100):
    df = pd.read_excel("Semantic_diversity.xlsx", skiprows=1)
    df['!term'] = df['!term'].str.lower()

    total_words = 0
    total_ambiguity_values = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['!term'] == word, 'SemD']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['!term'] == word_lemma, 'SemD']

                if not result.empty:
                    total_ambiguity_values += result.item()
                elif not result_lemma.empty:
                    total_ambiguity_values += result_lemma.item()
                else:
                    total_words -= 1
    return total_ambiguity_values / total_words * amount

def word_frequency(doc, amount=100):
    df = pd.read_excel("SUBTLEXusExcel2007.xlsx")
    df['Word'] = df['Word'].str.lower()

    total_words = 0
    total_wf_values = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['Word'] == word, 'Lg10WF']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['Word'] == word_lemma, 'Lg10WF']

                if not result.empty:
                    total_wf_values += result.item()
                elif not result_lemma.empty:
                    total_wf_values += result_lemma.item()
                else:
                    total_words -= 1
    return total_wf_values / total_words * amount

def word_prevelance(doc, amount=100):
    df = pd.read_excel("word_prevelance.xlsx")
    df['Word'] = df['Word'].str.lower()

    total_words = 0
    total_wp_values = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['Word'] == word, 'Prevalence']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['Word'] == word_lemma, 'Prevalence']

                if not result.empty:
                    total_wp_values += result.item()
                elif not result_lemma.empty:
                    total_wp_values += result_lemma.item()
                else:
                    total_words -= 1
    return total_wp_values / total_words * amount


def word_familiarity(doc, amount=100):
    df = pd.read_excel("word_prevelance.xlsx")
    df['Word'] = df['Word'].str.lower()

    total_words = 0
    total_word_freq_values = 0
    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['Word'] == word, 'Pknown']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['Word'] == word_lemma, 'Pknown']

                if not result.empty:
                    total_word_freq_values += result.item()
                elif not result_lemma.empty:
                    total_word_freq_values += result_lemma.item()
                else:
                    total_words -= 1
    return total_word_freq_values / total_words * amount

def age_of_aquisition(doc, amount=100):
    df = pd.read_excel("for_AoA.xlsx")
    df['Word'] = df['Word'].str.lower()

    total_words = 0
    total_AoA_values = 0

    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.pos_ == "NOUN":
                word = token.text.lower()
                result = df.loc[df['Word'] == word, 'AoA']
                word_lemma = token.lemma_.lower()
                result_lemma = df.loc[df['Word'] == word_lemma, 'AoA']

                if not result.empty:
                    total_AoA_values += result.item()
                elif not result_lemma.empty:
                    total_AoA_values += result_lemma.item()
                else:
                    total_words -= 1
    return total_AoA_values / total_words * amount

def frequency_nonwords(doc, amount=100):
    total_words = 0
    total_nonwords = 0

    word_set = set()
    with open("words_alpha.txt", "r") as f:
        for line in f:
            word_set.add(line.strip())


    for token in doc:
        if token.is_alpha:
            total_words += 1
            if token.text.lower() not in word_set and token.lemma_.lower() not in word_set:
                total_nonwords += 1

    print(f"total nonwords: {total_nonwords}")
    ratio_nonwords = total_nonwords / total_words * amount
    return ratio_nonwords

