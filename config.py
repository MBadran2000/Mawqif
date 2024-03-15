# TrainData_name = 'All targets-Train.csv'
# TestData_name = 'All targets-Test.csv'
TrainData_name = '/home/dr-nfs/m.badran/mawqif/All targets-Test.csv'
TestData_name = '/home/dr-nfs/m.badran/mawqif/All targets-Test.csv'

#selectedTarget = 'Covid Vaccine'
#selectedTarget = 'Digital Transformation'
# selectedTarget = 'Women empowerment'
selectedTarget = ['Women empowerment','Covid Vaccine','Digital Transformation']


#Ex_name = 'MARBERT_'+selectedTarget+'_Stance'
#Ex_name = 'AraBERT_'+selectedTarget+'_Stance'
#Ex_name = 'camelbert-da_'+selectedTarget+'_Stance'
# Ex_name = 'AraBERT-twitter_'+selectedTarget+'_Stance'

#"UBC-NLP/MARBERT
#"aubmindlab/bert-base-arabertv02"
#"CAMeL-Lab/bert-base-arabic-camelbert-da"
selectedModel = "aubmindlab/bert-base-arabertv02-twitter"

model_name = "bert-base-arabertv02"

MAX_TOKEN_COUNT = 128  ## Selection is based on the dataset graph 
N_EPOCHS = 1
BATCH_SIZE = 32
# BATCH_SIZE = 32

Modification = "None"
Version = "V1"

log_test = True

