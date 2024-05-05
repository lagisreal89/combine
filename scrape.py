import nfl_data_py as nfl #web scrapping for the combine data from 2000-2024
import pandas as pd
import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from sklearn import preprocessing
import re
import numpy as np
scraped_data = nfl.import_combine_data([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023])
scraped_data = scraped_data[['draft_year', 'draft_ovr', 'player_name','ht','wt','forty','bench','vertical','broad_jump']]
print(scraped_data.info())
draft = [] # tell whether or not someone was drafted (target value)
for row in scraped_data.itertuples(index=True):
   if (np.isnan(row.draft_ovr)):
        draft.append(False)
   else :
        draft.append(True)
scraped_data['drafted'] = draft
print(scraped_data.tail())