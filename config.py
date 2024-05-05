# TrainData_name = 'All targets-Train.csv'
# TestData_name = 'All targets-Test.csv'
TrainData_name = '/home/dr-nfs/m.badran/mawqif/All targets-Train.csv'
TestData_name = '/home/dr-nfs/m.badran/mawqif/All targets-Test.csv'


selectedTarget = [ 'Covid Vaccine', 'Women empowerment','Digital Transformation']
# selectedTarget = ['All']
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
WEIGHTED_SAMPLER = True
WEIGHT_DECAY = 0.001
LEARNING_RATE = 2e-5
USE_PEFT = 0 # 0:None, 1:LoRA, 2:LoHa, 3:LoKr, 4:AdaLoRA
LOSS = 0 # 0:CrossEntropyLoss, 1:LabelSmoothingCrossEntropyLoss, 2:FocalLoss
FREEZE_BERT = False
DROPOUT = 0.1
CONTRASTIVE_LOSS = 0 #0: None, 1:NTXentLoss, 2:AngularLoss, 3:ArcFaceLoss, 4:CircleLoss, 5:ContrastiveLoss, 6:CosFaceLoss
#7:DynamicSoftMarginLoss, 8:FastAPLoss, 9:GeneralizedLiftedStructureLoss, 10:InstanceLoss, 11:HistogramLoss. 12:IntraPairVarianceLoss
#13:TripletMarginLoss, 14:TupletMarginLoss, 15:SoftTripleLoss, 16:SphereFaceLoss
# 17:LiftedStructureLoss, 18:MarginLoss, 19:MultiSimilarityLoss, 20:NCALoss, 21:NormalizedSoftmaxLoss
# 22:PNPLoss, 23:ProxyAnchorLoss, 24:ProxyNCALoss, 25:SignalToNoiseRatioContrastiveLoss
CONTRASTIVE_LOSS_LAMBDA = 1
USE_ARABERT_PRE = True

USE_NLPAUG = True
NLPAUG_PROB = 0.1
taskWeights = [0.6,0.3,0.1]
# taskWeights = "Hierarchical Weighting"

### Modfiy the following after each major modification
Modification = "multi-task parallel v3.1 weight sampler v0"
Version = "V3.10_multitask"

log_test = True

