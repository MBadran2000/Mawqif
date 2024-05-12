### Getting Predictions:

1. **Install Dependencies:**  
   Ensure all dependencies listed in `requirements.txt` are installed.

2. **Download Models:**
   - Download the required models from [this link](https://drive.google.com/drive/folders/1lEF-5GXueAsmuafm4ocGVgYhXmNxFuTK?usp=drive_link) and place them in the `final_checkpoints` directory.

3. **Download Blind Dataset:**
   - Download the blind dataset and specify its path in `config.py` under the variable `BlindTestData_name`.

4. **Run Prediction Script:**  
   Execute `ensemble_prediction.py` to generate predictions.

### Training:

1. **Install Dependencies:**  
   Make sure to install all dependencies mentioned in `requirements.txt`.

2. **Configure Models:**
   - Adjust the details of the model in the configuration file.
   - Specify the paths for the training and testing datasets.

3. **Choose Training Approach:**
   - For multitask training, use `train_multitask.py`.
   - For multitarget training, use `train_multitarget.py`.
   - To employ a prompting task instead of classification, opt for `train_multiprompting.py`.
   - For standard classification training without multitask or multitarget, utilize `train.py`.
