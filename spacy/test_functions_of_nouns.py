import spacy
from pathlib import Path
from nlp_functions import abstractness
from nlp_functions import semantic_ambiguity
from nlp_functions import word_frequency, word_prevalence, word_familiarity, age_of_acquisition

print(f" automatic abstractness {abstractness(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/sample.txt")}")

nlp = spacy.load('en_core_web_lg')
doc = nlp(Path("sample_text/sample.txt").read_text(encoding='utf-8'))
noun_list = []
for token in doc:
    if token.pos_ == 'NOUN':
        noun_list.append(token.text.lower())
##############
#['life', 'things', 'town', 'countryside', 'hills', 'couple', 'farms', 'street', 'diner', 'name']
##############


#ABSTRACTNESS
manual =  {'life': 0.3717472118959108, 'thing': 0.31545741324921134, 'town': 0.21551724137931036, 'countryside': 0.2232142857142857, 'hill': 0.20283975659229211, 'couple': 0.2544529262086514, 'farm': 0.2178649237472767, 'street': 0.21052631578947367, 'diner': 0.20746887966804978, 'name': 0.2857142857142857}
list_of_values = []
for key, value in manual.items():
    list_of_values.append(value)
print(f" manual abstractness calculation: {sum(list_of_values) / len(list_of_values)}")


#SEMANTIC AMBIGUITY
list_of_semd_values = []
manual_semD_dict = {'life': 2.13, 'things': 1.95, 'town': 1.85,  'countryside': 1.55, 'hills': 1.61, 'couple': 1.87, 'farms': 1.27, 'street': 1.79, 'diner': 1.48, 'name': 2.05}
for key, value in manual_semD_dict.items():
    list_of_semd_values.append(value)

print(f" automatic semantic ambiguity {semantic_ambiguity(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/sample.txt")}")
print(f"manual semantic ambiguity calculation: {sum(list_of_semd_values) / len(list_of_semd_values)}")

#WORD FREQUENCY
list_of_wf_values = []
manual_wf_dict = {'life': 4.6088, 'things': 4.5482, 'town': 4.1019,  'countryside': 2.2577, 'hills': 3.0120, 'couple': 4.0567, 'farms': 2.1644, 'street': 3.8784, 'diner': 2.8084, 'name': 4.5150}
for key, value in manual_wf_dict.items():
    list_of_wf_values.append(value)
print(f" automatic word frequency {word_frequency(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/sample.txt")}")
print(f"manual word frequency calculation: {sum(list_of_wf_values) / len(list_of_wf_values)}")

#WORD PREVALENCE
list_of_wp_values = []
manual_wp_dict = {'life': 2.442, 'thing': 2.576, 'town': 2.439,  'countryside': 2.188, 'hill': 2.345, 'couple': 2.193, 'farm': 2.451, 'street': 2.427, 'diner': 2.443, 'name': 2.292}
for key, value in manual_wp_dict.items():
    list_of_wp_values.append(value)
print(f"automatic word prevalence {word_prevalence(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/sample.txt")}")
print(f"manual word prevalence calculation: {sum(list_of_wp_values) / len(list_of_wp_values)}")

#WORD FAMILIARITY
list_of_familiarity_values = []
manual_familiarity_dict = {'life': 1.00, 'thing': 1.00, 'town': 1.00,  'countryside': 0.99, 'hill': 1.00, 'couple': 0.99, 'farm': 1.00, 'street': 1.00, 'diner': 1.00, 'name': 0.99}
for key, value in manual_familiarity_dict.items():
    list_of_familiarity_values.append(value)
print(f"automatic word familiarity {word_familiarity(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/sample.txt")}")
print(f"manual word familiarity calculation: {sum(list_of_familiarity_values) / len(list_of_familiarity_values)}")

#AGE OF ACQUISITION
list_of_aoa_values = []
manual_aoa_dict = {'life': 5.6615, 'things': 4.31772, 'town': 5.47843,  'countryside': 8, 'hills': 4.49089, 'couple': 6.13383, 'farms': 4.51295, 'street': 4.60063, 'diner': 7.30251, 'name': 3.76751}
for key, value in manual_aoa_dict.items():
    list_of_aoa_values.append(value)
print(f"automatic age of acquisition {age_of_acquisition(nlp=spacy.load('en_core_web_lg'), file_path="sample_text/sample.txt")}")
print(f"manual age of acquisition calculation: {sum(list_of_aoa_values) / len(list_of_aoa_values)}")