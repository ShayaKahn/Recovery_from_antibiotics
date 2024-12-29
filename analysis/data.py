import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from utils.general_functions import *

### Recovery of gut microbiota of healthy adults following antibiotic exposure ###
os.chdir(r"C:\Users\USER\OneDrive\Desktop\Antibiotics\Recovery\Data")

#rel_abund_rarefied = pd.read_csv('annotated.mOTU.rel_abund.rarefied.tsv', sep='\t')
rel_abund_rarefied = pd.read_csv('annotated.mOTU.rel_abund.tsv', sep='\t')

rel_abund_rarefied = filter_data(rel_abund_rarefied)

baseline_columns = ['ERAS1_Dag0', 'ERAS2_Dag0', 'ERAS3_Dag0', 'ERAS4_Dag0', 'ERAS5_Dag0',
                    'ERAS6_Dag0', 'ERAS7_Dag0', 'ERAS8_Dag0', 'ERAS9_Dag0', 'ERAS10_Dag0',
                    'ERAS11_Dag0', 'ERAS12_Dag0']

baseline_columns_appear_4 = ['ERAS2_Dag0', 'ERAS3_Dag0', 'ERAS4_Dag0', 'ERAS5_Dag0',
                             'ERAS6_Dag0', 'ERAS7_Dag0', 'ERAS9_Dag0', 'ERAS11_Dag0', 'ERAS12_Dag0']

columns_4 = ['ERAS2_Dag4opt', 'ERAS3_Dag4', 'ERAS4_Dag4opt', 'ERAS5_Dag4', 'ERAS6_Dag4opt',
             'ERAS7_Dag4opt', 'ERAS9_Dag4', 'ERAS11_Dag4opt', 'ERAS12_Dag4opt']

columns_8 = ['ERAS1_Dag8',  'ERAS2_Dag8', 'ERAS3_Dag8', 'ERAS4_Dag8opt', 'ERAS5_Dag8', 'ERAS6_Dag8opt', 'ERAS7_Dag8',
             'ERAS8_Dag8', 'ERAS9_Dag8', 'ERAS10_Dag8', 'ERAS11_Dag8', 'ERAS12_Dag8']

columns_8_appear_4 = ['ERAS2_Dag8', 'ERAS3_Dag8', 'ERAS4_Dag8opt', 'ERAS5_Dag8', 'ERAS6_Dag8opt',
                        'ERAS7_Dag8', 'ERAS9_Dag8', 'ERAS11_Dag8', 'ERAS12_Dag8']

columns_42 = ['ERAS1_Dag42', 'ERAS2_Dag42', 'ERAS3_Dag42', 'ERAS4_Dag42', 'ERAS5_Dag42', 'ERAS6_Dag42',
              'ERAS7_Dag42',  'ERAS8_Dag42', 'ERAS9_Dag42', 'ERAS10_Dag42', 'ERAS11_Dag42', 'ERAS12_Dag42']

columns_42_appear_4 = ['ERAS2_Dag42', 'ERAS3_Dag42', 'ERAS4_Dag42', 'ERAS5_Dag42', 'ERAS6_Dag42',
                       'ERAS7_Dag42', 'ERAS9_Dag42', 'ERAS11_Dag42', 'ERAS12_Dag42']

columns_180 = ['ERAS1_Dag180', 'ERAS2_Dag180', 'ERAS3_Dag180', 'ERAS4_Dag180', 'ERAS5_Dag180', 'ERAS6_Dag180',
                'ERAS7_Dag180', 'ERAS8_Dag180', 'ERAS9_Dag180', 'ERAS10_Dag180', 'ERAS11_Dag180', 'ERAS12_Dag180']

columns_180_appear_4 = ['ERAS2_Dag180', 'ERAS3_Dag180', 'ERAS4_Dag180', 'ERAS5_Dag180', 'ERAS6_Dag180',
                        'ERAS7_Dag180', 'ERAS9_Dag180', 'ERAS11_Dag180', 'ERAS12_Dag180']

baseline_rel_abund_rarefied = rel_abund_rarefied[baseline_columns].values
baseline_rel_abund_rarefied = baseline_rel_abund_rarefied.T
baseline_rel_abund_rarefied = normalize_cohort(baseline_rel_abund_rarefied)

baseline_rel_abund_rarefied_appear_4 = rel_abund_rarefied[baseline_columns_appear_4].values
baseline_rel_abund_rarefied_appear_4 = baseline_rel_abund_rarefied_appear_4.T
baseline_rel_abund_rarefied_appear_4 = normalize_cohort(baseline_rel_abund_rarefied_appear_4)

rel_abund_rarefied_4 = rel_abund_rarefied[columns_4].values
rel_abund_rarefied_4 = rel_abund_rarefied_4.T
rel_abund_rarefied_4 = normalize_cohort(rel_abund_rarefied_4)

rel_abund_rarefied_8 = rel_abund_rarefied[columns_8].values
rel_abund_rarefied_8 = rel_abund_rarefied_8.T
rel_abund_rarefied_8 = normalize_cohort(rel_abund_rarefied_8)

rel_abund_rarefied_8_appear_4 = rel_abund_rarefied[columns_8_appear_4].values
rel_abund_rarefied_8_appear_4 = rel_abund_rarefied_8_appear_4.T
rel_abund_rarefied_8_appear_4 = normalize_cohort(rel_abund_rarefied_8_appear_4)

rel_abund_rarefied_42 = rel_abund_rarefied[columns_42].values
rel_abund_rarefied_42 = rel_abund_rarefied_42.T
rel_abund_rarefied_42 = normalize_cohort(rel_abund_rarefied_42)

rel_abund_rarefied_42_appear_4 = rel_abund_rarefied[columns_42_appear_4].values
rel_abund_rarefied_42_appear_4 = rel_abund_rarefied_42_appear_4.T
rel_abund_rarefied_42_appear_4 = normalize_cohort(rel_abund_rarefied_42_appear_4)

rel_abund_rarefied_180 = rel_abund_rarefied[columns_180].values
rel_abund_rarefied_180 = rel_abund_rarefied_180.T
rel_abund_rarefied_180 = normalize_cohort(rel_abund_rarefied_180)

rel_abund_rarefied_180_appear_4 = rel_abund_rarefied[columns_180_appear_4].values
rel_abund_rarefied_180_appear_4 = rel_abund_rarefied_180_appear_4.T
rel_abund_rarefied_180_appear_4 = normalize_cohort(rel_abund_rarefied_180_appear_4)
