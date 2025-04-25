Alzheimer's Disease Prediction Model
Overview
This repository contains a deep learning approach to predict Alzheimer's disease using sequential health data from the Health and Retirement Study (HRS). The project implements and compares two neural network architectures: Long Short-Term Memory (LSTM) and Temporal Convolutional Network (TCN) models.
Dataset
The analysis uses data from the Health and Retirement Study (HRS), a longitudinal panel study that surveys a representative sample of approximately 20,000 Americans over the age of 50 every two years. The dataset includes:

Cognitive assessments
Chronic health conditions (high blood pressure, diabetes, cancer, etc.)
Dental visit history
Alzheimer's disease diagnosis (target variable)

Features

Data preprocessing pipeline with missing value imputation and feature engineering
Balanced sampling using SMOTE to address class imbalance
Cross-validation with 5-fold stratified splits
Deep learning models (LSTM and TCN architectures)
Model evaluation with comprehensive metrics (accuracy, precision, recall, F1, AUC)
Statistical comparison of models using McNemar's test

Requirements
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
pyreadstat
imblearn
gdown
statsmodels
Model Architecture
LSTM Model
The LSTM model captures temporal dependencies in healthcare data:
pythonclass LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=0.6)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)
TCN Model
The TCN model uses dilated convolutions to process temporal sequences:
pythonclass TCNModel(nn.Module):
    def __init__(self, input_size, num_channels=[32, 32], kernel_size=2, dropout=0.6):
        super(TCNModel, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         stride=1, padding=(kernel_size-1)*dilation, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x[:, :, -1]
        out = self.fc(x)
        return self.sigmoid(out)
Results
The LSTM model significantly outperforms the TCN model for Alzheimer's disease prediction:
MetricLSTMTCNAccuracy0.99780.7901Precision0.99530.6197Recall0.99810.9582F1 Score0.99670.7527AUC0.99990.9231
McNemar's test confirms that the difference between the models is statistically significant (p < 0.05).
Usage

Clone the repository
Install the required dependencies
Run the main script to download data and train models:

python train_model.py
Future Work

Explore additional neural network architectures
Implement interpretability techniques to identify key predictive features
Extend the analysis to predict early-stage cognitive decline
Develop a risk stratification system for personalized interventions

License
MIT License
Citation
If you use this code in your research, please cite:
@misc{alzheimers_prediction_hrs,
  author = {Niloofar Deravi},
  title = {Alzheimer's Disease Prediction Using Sequential Health Data},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/niloofarderavi/LSTM-Vs.-TCN-for-Alzheimer-prediction-based-on-dental-care}}