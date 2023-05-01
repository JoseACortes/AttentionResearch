# this file has the code to process the raw data and create the data files
# it includes the following functions:
# 1. outlierHandling: this function handles the outliers in the data
# 2. tansformData: this function transforms the data to make it suitable for the model
# 3. untransformData: this function transforms the data back to the original form after the model has been run
import pandas as pd
import numpy as np

def standardize(x, mean, std):
    return (x-mean)/std
def unstardardize(x, mean, std):
    return x*std+mean

# mean and standard deviation for the reaction time after log transformation
rtmean = 5.909439791104729
rtstd = 0.20024553121497682

def standardizeRT(x):
    return standardize(x, rtmean, rtstd)
def processRT(x):
    return standardizeRT(np.log(x))
def unstandardizeRT(x):
    return unstardardize(x, rtmean, rtstd)
def unprocessRT(x):
    return np.exp(unstandardizeRT(x))

# mean and standard deviation for right pupil dialation after log transformation
rpmean = 1.0298308234047164
rpstd = 0.1408170702292441

def standardizeRP(x):
    return standardize(x, rpmean, rpstd)
def processRP(x):
    return standardizeRP(np.log(x))
def unstandardizeRP(x):
    return unstardardize(x, rpmean, rpstd)
def unprocessRP(x):
    return np.exp(unstandardizeRP(x))

# mean and standard deviation for left pupil dialation after log transformation
lpmean = 1.024794582075571
lpstd = 0.14519670907378032

def standardizeLP(x):
    return standardize(x, lpmean, lpstd)
def processLP(x):
    return standardizeLP(np.log(x))
def unstandardizeLP(x):
    return unstardardize(x, lpmean, lpstd)
def unprocessLP(x):
    return np.exp(unstandardizeLP(x))

def cutGaze(x):
    if x > 1 or x < 0:
        return np.nan
    else:
        return x
    
rgxmean = 0.5089461
rgxstd = 0.07544981224141276

def standardizeRGX(x):
    return standardize(x, rgxmean, rgxstd)
def processRGX(x):
    return standardizeRGX(cutGaze(x))
def unstandardizeRGX(x):
    return unstardardize(x, rgxmean, rgxstd)
def unprocessRGX(x):
    return unstandardizeRGX(x)

rgymean = 0.4495476
rgystd = 0.06781282432116541

def standardizeRGY(x):
    return standardize(x, rgymean, rgystd)
def processRGY(x):
    return standardizeRGY(cutGaze(x))
def unstandardizeRGY(x):
    return unstardardize(x, rgymean, rgystd)
def unprocessRGY(x):
    return unstandardizeRGY(x)

lgxmean = 0.5089461
lgxstd = 0.07544981224141276

def standardizeLGX(x):
    return standardize(x, lgxmean, lgxstd)
def processLGX(x):
    return standardizeLGX(cutGaze(x))
def unstandardizeLGX(x):
    return unstardardize(x, lgxmean, lgxstd)
def unprocessLGX(x):
    return unstandardizeLGX(x)

lgymean = 0.44419505000000004
lgystd = 0.06809138358348428

def standardizeLGY(x):
    return standardize(x, lgymean, lgystd)
def processLGY(x):
    return standardizeLGY(cutGaze(x))
def unstandardizeLGY(x):
    return unstardardize(x, lgymean, lgystd)
def unprocessLGY(x):
    return unstandardizeLGY(x)


def process(dataframe):
    outframe = dataframe.copy()
    outframe['right_pupil'] = outframe['right_pupil'].apply(processRP)
    outframe['left_pupil'] = outframe['left_pupil'].apply(processLP)
    outframe['right_gaze_x'] = outframe['right_gaze_x'].apply(processRGX)
    outframe['right_gaze_y'] = outframe['right_gaze_y'].apply(processRGY)
    outframe['left_gaze_x'] = outframe['left_gaze_x'].apply(processLGX)
    outframe['left_gaze_y'] = outframe['left_gaze_y'].apply(processLGY)
    
    outframe['rt'] = outframe['rt'].apply(processRT)
    return outframe

def unprocess(dataframe):
    outframe = dataframe.copy()
    outframe['right_pupil'] = outframe['right_pupil'].apply(unprocessRP)
    outframe['left_pupil'] = outframe['left_pupil'].apply(unprocessLP)
    outframe['right_gaze_x'] = outframe['right_gaze_x'].apply(unprocessRGX)
    outframe['right_gaze_y'] = outframe['right_gaze_y'].apply(unprocessRGY)
    outframe['left_gaze_x'] = outframe['left_gaze_x'].apply(unprocessLGX)
    outframe['left_gaze_y'] = outframe['left_gaze_y'].apply(unprocessLGY)
    
    outframe['rt'] = outframe['rt'].apply(unprocessRT)
    return outframe