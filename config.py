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
# selectedModel = "aubmindlab/aragpt2-base"

# model_name = "bert-base-arabertv02"

MAX_TOKEN_COUNT = 128  ## Selection is based on the dataset graph 
N_EPOCHS = 50#40
BATCH_SIZE = 32#32

WEIGHTED_LOSS = False
WEIGHTED_SAMPLER = False
WEIGHT_DECAY = 0.001
LEARNING_RATE = 2e-5
USE_PEFT = 0 # 0:None, 1:LoRA, 2:LoHa, 3:LoKr, 4:AdaLoRA
LOSS = 0 # 0:CrossEntropyLoss, 1:LabelSmoothingCrossEntropyLoss, 2:FocalLoss
FREEZE_BERT = False
DROPOUT = 0.1
CONTRASTIVE_LOSS = 1 #0: None, 1:NTXentLoss

### Modfiy the following after each major modification
Modification = "Contrastive Loss NTXentLoss"
Version = "V7.1"
log_test = False