# Importing data

import pandas as pd
import os
print (f'Starting data import ⏳')
current_directory = os.path.realpath('data.py')
parent_directory = parent_directory = os.path.dirname(current_directory)

class ImportData:

    def __ini__(self):
        self.current_directory = current_directory
        self.parent_directory = parent_directory

    def abbreviations():

        data_abbreviations = pd.read_csv(os.path.join(parent_directory,'data_raw','Abbreviations_and_Slang.csv'))

        return data_abbreviations

    def slangs():

        data_slangs = pd.read_csv(os.path.join(parent_directory,'data_raw','slang.csv'))[['acronym', 'expansion']]

        return data_slangs
