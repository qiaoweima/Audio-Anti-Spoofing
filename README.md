# Audio-Anti-Spoofing
A pytorch implementation of "A Lightweight and Efficient Model for Audio Anti-Spoofing"
# DataSet
ASVspoof 2019 LA & ASVspoof 2015
# Usage
### Data Preparation
```
python ASVspoof15&19_LA_Data_Preparation.py 
```
It generates audio of fixed length from the given dataset.
### Training
```
python train.py
```
We use focal loss to train the model, which solves the problem of difficult classification samples to some extent. The training records are saved in log files, and the model files are saved every epoch of training.
