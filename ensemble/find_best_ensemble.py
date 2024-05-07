import csv,os
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
import pandas as pd
from itertools import combinations
from collections import Counter
from MyStanceEval_Final import log_StanceEval
import shutil

def find_max(row):
    # List = [row['Label1'],row['Label2'], row['Label3'], row['Label4'], row['Label5']]
    List = [row['Label1'],row['Label2'], row['Label3'], row['Label4'], row['Label5'], row['Label6'], row['Label7']]#7

    # print(List, max((List), key = List.count))
    return max((List), key = List.count)


df = pd.read_csv('results/combined_data.csv')

results = [f for f in df.columns if 'Label' in f and not 'gt' in f]
print(results,len(results))

# df['Label'] = df['Label_gt'].copy()
# combinations_list = list(combinations(results, 5))
combinations_list = list(combinations(results, 7))#7

# combinations_list = combinations_list[:1000]
print(len(combinations_list))

outpath = '/home/dr-nfs/m.badran/mawqif/results/ensembles/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

best_comb = 0
best_r = 0
for i in range(len(combinations_list)):
    print(i)


    df['Label1'] = df[combinations_list[i][0]].copy()
    df['Label2'] = df[combinations_list[i][1]].copy()
    df['Label3'] = df[combinations_list[i][2]].copy()
    df['Label4'] = df[combinations_list[i][3]].copy()
    df['Label5'] = df[combinations_list[i][4]].copy()
    df['Label6'] = df[combinations_list[i][5]].copy()#7
    df['Label7'] = df[combinations_list[i][6]].copy()#7

    df['Label_ensemble'] = df.apply(find_max, axis=1)
    # print(combinations_list[i])
    # comb = combinations_list[i][0][11:]+combinations_list[i][1][10:]+combinations_list[i][2][10:]+combinations_list[i][3][10:]+combinations_list[i][4][10:]
    # comb = combinations_list[i][0][11:]+combinations_list[i][1][10:]+combinations_list[i][2][10:]+combinations_list[i][3][10:]+combinations_list[i][4][10:]+combinations_list[i][5][10:]+combinations_list[i][6][10:]
    # comb = combinations_list[i][0][8:]+"-"+combinations_list[i][1][8:]+"-"+combinations_list[i][2][8:]+"-"+combinations_list[i][3][8:]+"-"+combinations_list[i][4][8:]
    comb = combinations_list[i][0][8:]+"-"+combinations_list[i][1][8:]+"-"+combinations_list[i][2][8:]+"-"+combinations_list[i][3][8:]+"-"+combinations_list[i][4][8:]+"-"+combinations_list[i][5][8:]+"-"+combinations_list[i][6][8:]



    # print(comb)

    desired_columns_order = ['Index', 'Category', 'Text', 'Label_ensemble']

    # # Read the CSV file
    # with open('input.csv', 'r') as csv_file:
    #     reader = csv.DictReader(csv_file)
    #     rows = list(reader)

    # Write data to a text file without column names and in the desired order
    with open(outpath+comb+'.txt', 'w') as txt_file:
        for index, row in df.iterrows():
            # Rearrange the data according to desired order
            reordered_row = [str(row[column]) for column in desired_columns_order]
            # Write row to the text file
            txt_file.write('\t'.join(reordered_row) + '\n')
    # logger = CometLogger(
    #     experiment_name=comb ,
    #     api_key='jFe8sB6aGNDL7p2LHe9V99VNy',
    #     workspace='mbadran2000',  # Optional
    #     #  save_dir='lightning_logs',  # Optional
    #     project_name='Mawqif-Ensemble',  # Optional
    # )    
    logger = None
    r = log_StanceEval(outpath+comb+'.txt',logger,"test")
    if r > best_r:
        best_comb = comb
        best_r = r
    print(best_comb,best_r)
    if i % 10000==0:
        shutil.rmtree('/home/dr-nfs/m.badran/mawqif/results/ensembles/')
        os.makedirs(outpath)


###101
## 11.14.13.9.19 0.8302006553887065
## 20.13.15.3.5 0.8395059369859195
## 11.14.2.13.25.3.5 0.8365767891733066

## 01.11-04.3-04.5-01.25-01.17-01.15-04.12 0.8380861288469984
## 01.8-01.20-01.1-04.13-04.5-01.25-04.18 0.838996121256404
#01.8-04.13-04.5-01.25-01.15 0.8432030970203455
