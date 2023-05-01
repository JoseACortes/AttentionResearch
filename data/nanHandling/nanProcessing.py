import pandas as pd
import numpy as np

def nanPercentages(dataframe):
    nancounts = dataframe.query('currentobject == "Zeros"').isna().groupby('st').sum()[['right_pupil', 'left_pupil', 'right_gaze_x', 'right_gaze_y', 'left_gaze_x', 'left_gaze_y']]
    hitcounts = dataframe.groupby('st').count()[['right_pupil', 'left_pupil', 'right_gaze_x', 'right_gaze_y', 'left_gaze_x', 'left_gaze_y']]
    percentnan = nancounts/(hitcounts+nancounts)
    percentnan['whole'] = percentnan.apply(lambda x: x.mean(), axis=1)
    percentnan['max'] = percentnan.apply(lambda x: x.max(), axis=1)
    return percentnan

def nan_to_0(X):
    X = np.array(X)
    X = np.nan_to_num(X, 0)
    return X

def nan_to_avg(X):
    X = np.array(X)
    u = np.nanmean(X)
    X = np.nan_to_num(X, nan=u)
    return X

def nan_to_interp(X):
    X = np.array(X)
    nanmask = ~(np.isnan(X))
    X = np.interp(np.arange(len(X)), np.arange(len(X))[nanmask], X[nanmask])
    return X