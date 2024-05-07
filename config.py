# TrainData_name = 'All targets-Train.csv'
# TestData_name = 'All targets-Test.csv'
TrainData_name = '/home/dr-nfs/m.badran/mawqif/All targets-Train.csv'
TestData_name = '/home/dr-nfs/m.badran/mawqif/All targets-Test.csv'
BlindTestData_name = '/home/dr-nfs/m.badran/mawqif/Mawqif_AllTargets_Blind Test.csv'

selectedTarget = [ 'Covid Vaccine', 'Women empowerment','Digital Transformation']
# selectedTarget = ['All'] # for multi-target approch
# selectedTarget = [ 'Covid Vaccine']


# selectedModel = "aubmindlab/bert-base-arabertv02-twitter"


selectedModels = {'Women empowerment':"aubmindlab/bert-base-arabertv02-twitter",'Covid Vaccine':'moha/arabert_c19', 'Digital Transformation':"aubmindlab/bert-base-arabertv02-twitter"}


MAX_TOKEN_COUNT = 128   

N_EPOCHS = 60
BATCH_SIZE = 32

WEIGHTED_LOSS = True
WEIGHTED_SAMPLER = False
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
# 22:PNPLoss, 23:ProxyAnchorLoss, 24:ProxyNCALoss, 25:SignalToNoiseRatioContrastiveLoss #26:NTXentLoss temperature=0.5 
CONTRASTIVE_LOSS_LAMBDA = 1
USE_ARABERT_PRE = False

USE_NLPAUG = False
NLPAUG_PROB = 0.1
taskWeights = [0.7,0.2,0.1]

Modification = "multi-task parallel (Weighted Loss for all tasks) without arabert preprocessor and without nlp aug"
Version = "V104." +str(CONTRASTIVE_LOSS)

log_test = True

