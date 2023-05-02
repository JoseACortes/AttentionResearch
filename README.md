# Predicting Reaction Times from Pupilometry Data
-----------------
## Introduction
-----------------
## Objective
-----------------
## Data
-----------------
## Methods
-----------------
## Results
-----------------
## Repository
```
Project Structure:
├── code/ (this repository)
│   ├── data/
│       ├── processing/
│           ├── processData.py
│           ├── processData.ipynb
│       ├── visualizations/
│           ├── visualizations.py
│           ├── visualizations.ipynb
│       ├── nanHandling/
│           ├── nanHandling.py
│           ├── nanHandling.ipynb
│   ├── models/
│       ├── RandomBaseline.py
│       ├── FullyConnectedNeuralNetwork.py
│       ├── VisualHeatmap.py
│       ├── ConvolutedNeuralNetwork.py
│       ├── LongShortTermMemoryNetwork.py
│       ├── TrainingAndTesting.ipynb
│   ├── extra/
│   ├── readme.md
├── data/
│   ├── raw/
│       ├── rawEyePupilometryData.csv
│       ├── subjectInfo.csv
│   ├── processed/
│       ├── processedData.pkl
│       ├── processedSubjectInfo.pkl
│       ├── TrainingData.pkl
│       ├── TrainingSubjectInfo.pkl
│       ├── TestingData.pkl
│       ├── TestingSubjectInfo.pkl
```

### Code (This Repository)

### Data

### Processing

### Visualization

Notebooks for visualizing the data.

### Models
5 models were used to predict reaction times. Each model takes all pupilometry features. The models are listed below.

**Model 1: Random Baseline**  
-> In this model, we randomly assign a reaction time to each trial. The reaction time is randomly selected from the distribution of reaction times in the data.

**Model 2: Full Connected Neural Network**  
-> In this model, we use a fully connected neural network to predict reaction times. The input is a vector of concatenated features.

**Model 3: Visual Heatmap**  
-> In this model, we use the visual heatmap to predict reaction times. The visual heatmap is a 2D matrix of eye position values. The matrix is 100x100, and each cell represents a 1x1 degree area of the visual field. The values in the matrix are the average pupil dilation values for each cell.

**Model 4: Convoluted Neural Network**  
-> In this model, we use a convolutional neural network to predict reaction times. The input is a heatmap of concatenated features shaped Features x Window Length.

**Model 5: Long Short-Term Memory Network**  
-> In this model, we use a long short-term memory network to predict reaction times. The input is the time series of each feature.

### Full Data
This is the full data used in the project. The data is split into raw data and processed data.  
The Data is 2.2GB.

-----------------
