import os
from utils.prediction_utils_multitask import get_predictions, predict
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def save_pred_gt(Ex_name, target, data_texts, data_pred, data_true, val_or_test):
    mapping = {0:'NONE', 1:'FAVOR', 2:'AGAINST'}
    data_pred_1 = [mapping[t.item()] for t in data_pred]
    data_true_1 = [mapping[t.item()] for t in data_true]

    folder_path ='-'.join(("checkpoints/"+Ex_name).split("-")[:-1]) 
    file_path_pred_File = folder_path+ "/"+val_or_test+"_pred_File.txt"
    file_path_gt_file = folder_path+ "/"+val_or_test+"_gt_file.txt"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(file_path_pred_File, "a") as file:
        num_lines = 0
    with open(file_path_pred_File, "r") as file:
        lines = file.readlines()
        num_lines = len(lines)
    for i in range(len(data_pred_1)):
        with open(file_path_pred_File, "a") as file:
            file.write(f"{num_lines}\t{target}\t{data_texts[i]}\t{data_pred_1[i]}\n")
        with open(file_path_gt_file, "a") as file:
            file.write(f"{num_lines}\t{target}\t{data_texts[i]}\t{data_true_1[i]}\n")
        num_lines+=1

def log_results(Ex_name, trained_model, logger, selectedTarget,dataset, val_or_test):
    data_texts, data_pred, data_pred_probs, data_true = get_predictions(
      trained_model,
      dataset
    )

    class_names = ['None','Favor','Against']
    report_val = classification_report(data_true, data_pred, target_names=class_names, zero_division=0, digits=4, output_dict=True)
    save_pred_gt(Ex_name, selectedTarget, data_texts, data_pred, data_true, val_or_test)

    for key, value in report_val.items():
        if key == "accuracy":
            logger.experiment.log_metric(val_or_test+"_"+key, value)
        else:
            logger.experiment.log_metrics(value,prefix=val_or_test+"_"+key)
    logger.experiment.log_confusion_matrix(
        data_true.tolist(),
        data_pred.tolist(),
        title=val_or_test+" Confusion Matrix",
        file_name=val_or_test+"-confusion-matrix.json"
    )

