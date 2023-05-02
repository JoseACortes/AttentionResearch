import pandas as pd
import numpy as np
from processing.processData import process
from nanHandling.nanProcessing import nanPercentages
from nanHandling.nanProcessing import nan_to_interp
from alive_progress import alive_bar

trainingData = pd.read_pickle('../../fulldata/processed/trainingData.pkl')
trainingSubjectInfo = pd.read_pickle('../../fulldata/processed/trainingSubjectInfo.pkl')
testingData = pd.read_pickle('../../fulldata/processed/testingData.pkl')
testingSubjectInfo = pd.read_pickle('../../fulldata/processed/testingSubjectInfo.pkl')

def deepProcess(dataframe, subjectframe, nancut = .4):
    print('Copying dataframes...')
    dataframe = dataframe.copy()
    subjectframe = subjectframe.copy()
    # 1. Process data
    print('Processing data...')
    dataframe = process(dataframe)
    # 2. Remove NaN-heavy trials
    print('Removing NaN-heavy trials')
    percentnan = nanPercentages(dataframe)
    subjectframe['percentNanWhole'] = percentnan['whole']
    subjectframe['percentNanMax'] = percentnan['max']
    nancutSubjectInfo = subjectframe.query('percentNanWhole < .4 and percentNanMax < .6')
    nancutData = dataframe.loc[nancutSubjectInfo.index]
    nancutData.sort_values(by=['time'], inplace=True)
    # 3. Interpolate NaNs
    print('Interpolating NaNs...')
    sts = nancutData.index.unique().to_list()
    with alive_bar(len(sts), force_tty=True) as bar:
        for st in sts:
            nancutData.loc[[st], ['right_pupil', 'left_pupil', 'right_gaze_x', 'right_gaze_y', 'left_gaze_x', 'left_gaze_y']] = nancutData.loc[[st], ['right_pupil', 'left_pupil', 'right_gaze_x', 'right_gaze_y', 'left_gaze_x', 'left_gaze_y']].apply(nan_to_interp)
            bar()
    return nancutData

# X_y_train = deepProcess(trainingData, trainingSubjectInfo)
# X_y_train.to_pickle('../../fulldata/processed/X_y_train.pkl')

# X_y_test = deepProcess(testingData, testingSubjectInfo)
# X_y_test.to_pickle('../../fulldata/processed/X_y_test.pkl')