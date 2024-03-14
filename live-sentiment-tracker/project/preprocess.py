import numpy as np
import pandas as pd
import string
from data import ImportData
import warnings


warnings.filterwarnings("ignore")

# Import data from data.py

data_abbreviations = ImportData.abbreviations()
data_slangs = ImportData.slangs()

print(f'Abbrevation shape: {data_abbreviations.shape}')
print(f'Slangs shape: {data_slangs.shape}')

assert data_abbreviations.shape == (114, 2)
assert data_slangs.shape == (3357, 2)

print(f'Data imported correctly ✅')


print (f'Starting preprocess ⏳')


data_abbreviations = ImportData.abbreviations()
data_slangs = ImportData.slangs()

remove_list = [13, 15, 16, 27, 41, 63, 66, 67, 112]
add_list = [159, 160, 185, 330, 2362]


data_slangs.rename(columns = {'acronym':'Abbreviations','expansion':'Text'}, inplace = True)

data_slang_all = pd.concat([data_abbreviations , data_slangs], axis=0)

# Concatenating slang and abbreviation datasets
data_slangs.rename(columns = {'acronym':'Abbreviations','expansion':'Text'}, inplace = True)
data_slangs = data_slangs[data_slangs.index.isin(add_list)]
data_abbreviations.drop(axis=0, labels=remove_list, inplace=True)
data_slang_all = pd.concat([data_abbreviations , data_slangs], axis=0)

# Drop duplicates and null values
data_slang_all.drop_duplicates(inplace=True)
data_slang_all.dropna(inplace=True)

# Checking
print(f'Data + Slang full dataset shape: {data_slang_all.shape}')
print(f'Data + Slang full dataset null: {data_slang_all.isnull().sum()[0]}')
print(f'Data + Slang full dataset duplicates: {data_slang_all.duplicated().sum()}')

# Transforming DF into dict for mapping
slang_dict = dict(zip(data_slang_all.Abbreviations, data_slang_all.Text))
data_abbreviations_dict = dict(zip(data_abbreviations.Abbreviations, data_abbreviations.Text))


assert data_slang_all.shape == (108, 2)
assert len(slang_dict) == 108

print(f'Preprocess done correctly ✅')


print (f'Creating slang translator ⏳')

# Creating class for slang
class SlangTranslation:

  def __init__(self, sentence):

    self.sentence = sentence

  def remove_punctuation(self, sentence):
    """Iterates through each word of the string and removes punctuation"""
    for punctuation in string.punctuation:
        sentence = sentence.replace(punctuation, '')

    return sentence

  def string_translator(self, sentence):
    """Iterates through each word of the string and translates them"""

    sentence = ' '.join([slang_dict.get(i, i) for i in sentence.split()])

    return sentence

  def apply_translator(self, sentence):
    """Takes the text column as input, outputs the same column translated."""

    sentence = self.remove_punctuation(sentence)

    sentence = self.string_translator(sentence)

    return sentence

print(f'Slang translator created ✅')
