# Saurabh Annadate
## Text Analytics - Homework3


### Data Download

**Link to the data:** https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M 

**File Name:** _amazon_review_full_csv.tar.gz_

**Related Paper Link:** https://arxiv.org/pdf/1502.01710.pdf

__Instructions:__
1. After cloning the repo, download the above tar.gz file and paste it in the **Data/** folder
2. Unzip the file
3. The unzipped folder should contain three files:
    * train.csv
    * test.csv
    * readme.txt

<br>

### Running Data Summary
```
python run.py run_dataset_stats
```
The data summaries will appear in the **Outputs/** folder

<br>

### Running Logistic Regression
```
python run.py run_dataset_stats
```
The summary will appear in **Outputs/** and the model will be saved in **Models/**

<br>

### Running SVM
```
python run.py run_svm
```
The summary will appear in **Outputs/** and the model will be saved in **Models/**

<br>

### Running fasttext
```
python run.py run_fasttext
```
The summaries will appear in **Outputs/**

<br>

### Running CNN
```
python run.py run_cnn
```
The model will be saved in **Models/**

<br>

### Predicting using SVM
Enter your input in **Tests/Text.txt** and execute the following command:

```
python run.py predict_svm
```
The output will appear in **Outputs/svm_result.json**

<br>

### Predicting using CNN
Enter your input in **Tests/Text.txt** and execute the following command:

```
python run.py predict_cnn
```
The output will appear in **Outputs/cnn_result.json**

