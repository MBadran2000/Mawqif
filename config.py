# TrainData_name = 'All targets-Train.csv'
# TestData_name = 'All targets-Test.csv'
TrainData_name = '/home/dr-nfs/m.badran/mawqif/All targets-Train.csv'
TestData_name = '/home/dr-nfs/m.badran/mawqif/All targets-Test.csv'


selectedTarget = [ 'Covid Vaccine', 'Women empowerment','Digital Transformation']
# selectedTarget = [ 'Covid Vaccine']

#Ex_name = 'MARBERT_'+selectedTarget+'_Stance'
#Ex_name = 'AraBERT_'+selectedTarget+'_Stance'
#Ex_name = 'camelbert-da_'+selectedTarget+'_Stance'
# Ex_name = 'AraBERT-twitter_'+selectedTarget+'_Stance'

# selectedModel = "UBC-NLP/MARBERT"
# selectedModel =  "aubmindlab/bert-base-arabertv02"
# selectedModel = "CAMeL-Lab/bert-base-arabic-camelbert-da"
selectedModel = "aubmindlab/bert-base-arabertv02-twitter"
# selectedModel = 'aubmindlab/bert-large-arabertv02-twitter' ## no enough memory 
# selectedModel = 'CAMeL-Lab/bert-base-arabic-camelbert-msa-did-madar-twitter5'
# selectedModel = 'CAMeL-Lab/bert-base-arabic-camelbert-mix'
# selectedModel =  'CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment'
# selectedModel = 'moha/arabert_c19'

# model_name = "bert-base-arabertv02"

MAX_TOKEN_COUNT = 128  ## Selection is based on the dataset graph 
N_EPOCHS = 40
BATCH_SIZE = 32

WEIGHTED_LOSS = False
WEIGHT_DECAY = 0.001
LEARNING_RATE = 2e-5

### Modfiy the following after each major modification
Modification = "None"
Version = "V2.2"
log_test = False