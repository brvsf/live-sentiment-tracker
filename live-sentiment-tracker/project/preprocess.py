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

remove_list = [
    25, 35, 65, 63, 0, 165, 1982,
    75, 1, 2495, 90, 216, 11, 251,
    106, 5, 279, 12, 311, 368, 24,
    437, 36, 453, 476, 50, 512, 611,
    621, 43, 78, 684, 721, 744, 100,
    102, 747, 810, 53, 17, 888, 120,
    941, 984, 1023, 62, 1100, 141,
    1237, 171, 1712, 1727, 1732, 173,
    1767, 175, 1845, 178, 1934, 75,
    1982, 2062, 2134, 78, 188, 2138,
    2198, 82, 83, 2215, 196, 2230,
    84, 2236, 2237, 2332, 2392, 233,
    2753, 3019, 3158, 3159, 3160,
    3227, 3279, 979, 138, 2633, 16,
    42, 44, 67, 128, 157, 158, 241,
    305, 306, 315, 342, 381, 388,
    392, 394, 413, 452, 456, 526,
    560, 561, 599, 662, 669, 727,
    768, 789, 838, 954, 963, 964,
    992, 1011, 1076, 1183, 1278,
    1303, 1359, 1366, 1372, 1429,
    1463, 1462, 1483, 1493, 1569,
    1662, 1693, 1737, 1741, 1742,
    1747, 1754, 1769, 1785, 1804,
    1822, 1829, 1858, 1869, 2002,
    2051, 2114, 2153, 2183, 2203,
    2287, 2372, 2391, 2406, 2415,
    2421, 2422, 2437, 2492, 2513,
    2558, 2565, 2718, 2835, 2836,
    2850, 2930, 2960, 3026, 3027,
    3085, 3116, 3350, 101, 156,
    193, 297, 303, 357, 367, 401,
    481, 518, 531, 551, 567, 601,
    631, 720, 728, 759, 764, 880,
    889, 895, 901, 913, 939, 1108,
    1277, 1276, 1306, 1362, 1371,
    1529, 1541, 1698, 1815, 1892,
    1919, 1938, 1957, 1961, 2015,
    2071, 2100, 2106, 2126, 2142,
    2146, 2184, 2186, 2239, 2249,
    2327, 2363, 2375, 2439, 2443,
    2482, 2487, 2493, 3311, 3306,
    3239, 3202, 3194, 3148, 3142,
    3100, 2962, 2961, 2944, 2939,
    2940, 2904, 2851, 2844, 2833,
    2755, 2720, 2714, 2704, 2456,
    2697, 2687, 2674, 2675,2673 ,
    2619, 2592, 2589, 2584, 2568,
    2566, 2526, 2453, 2455, 13, 15,
    55, 71, 99, 174, 179, 198, 288,
    290, 448, 654, 656, 678, 742,
]

data_slangs.rename(columns = {'acronym':'Abbreviations','expansion':'Text'}, inplace = True)

data_slang_all = pd.concat([data_abbreviations , data_slangs], axis=0)


# Drop duplicates and null values
data_slang_all.drop_duplicates(inplace=True)
data_slang_all.dropna

# Drop significant rows
data_slang_all.drop(axis=0, labels=remove_list, inplace=True)

# Checking
print(f'Data + Slang full dataset shape: {data_slang_all.shape}')
print(f'Data + Slang full dataset null: {data_slang_all.isnull().sum()[0]}')
print(f'Data + Slang full dataset duplicates: {data_slang_all.duplicated().sum()}')


# Transforming DF into dict for mapping
slang_dict = dict(zip(data_slang_all.Abbreviations, data_slang_all.Text))
data_abbreviations_dict = dict(zip(data_abbreviations.Abbreviations, data_abbreviations.Text))

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
