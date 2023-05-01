import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def trialFeatureVisual(data, subject, trial, feature, figsize=(15, 5)):
    """A function to visualize a specific feature of a specific trial of a specific subject.

    Args:
        data (dataframe): a dataframe of index: timestep, columns: features and rt, subject, trial, currentobject
        subject (int): specific subject to visualize
        trial (int): specific trial to visualize
        feature (str): feature to visualize
        figsize (tuple, optional): size of figure. Defaults to (15, 5).
    """
    # cut data to subject and trial
    section = data.query('subject == {} and trial == {}'.format(subject, trial))
    section = section.sort_values('time')
    # get times for the start of each trial phase
    phases = section.groupby('currentobject').first()
    phases = list(zip(phases.index, phases['time'].values))
    
    plt.figure(figsize=figsize)
    plt.title('Subject {}, Trial {}, {}'.format(subject, trial, feature))
    sns.scatterplot(x='time', y=feature, data=section)
    
    for phase, phasetime in phases:
        plt.axvline(x=phasetime, color='grey', linestyle='--')
        plt.text(phasetime+20, 1, phase)
        
    start = None
    end = None
    for phase in phases:
        if phase[0] == 'Timer':
            start = phase[1]
        elif phase[0] == 'Feedback':
            end = phase[1]
    
    if ('Timer' in [phase[0] for phase in phases]) and ('Feedback' in [phase[0] for phase in phases]):
        plt.hlines(y=-1, xmin=start, xmax=end, color='black', linestyle='--')
        plt.text(start+20, -1.3, 'RT: {}'.format(str(section['rt'].values[0])[:5]))
    
    plt.ylim(-2, 2)
    plt.xticks(np.arange(0, 16, 2)*1000+section['time'].min(), labels=np.arange(0, 16, 2))
    plt.show()

def trialFeaturePairplot(data, subject, trial, features='all', phases='all', hue=None):
    """A function to visualize a pairplot of a features and reaction times.
    
    Args:
        data (_dataframe_): a dataframe of index: timestep, columns: features and rt, subject, trial, currentobject.
        subject (int): specific subject to visualize.
        trial (int): specific trial to visualize.
        features (str or list, optional): feature(s) to visualize. Defaults to 'all'.
        phases (str or list, optional): phase(s) to visualize. Defaults to 'all'.
        hue (str, optional): feature to color by "time" or "currentobject". Defaults to None.
    """
    section = data.query('subject == {} and trial == {}'.format(subject, trial)).copy()
    
    phasestring = 'All Phases'
    if phases == 'all':
        pass
    else:
        section = section.query('currentobject in "{}"'.format(phases))
        phasestring = 'Phase(s): {}'.format(phases)

    featurestring = 'All Features'    
    if features == 'all':
        features = ['right_gaze_x', 'right_gaze_y', 'left_gaze_x', 'left_gaze_y', 'right_pupil', 'left_pupil']
    else:
        if isinstance(features, str):
            features = [features]
        featurestring = 'Feature(s):'+', '.join(features)
    
    huewrap = []
    huetitle = ''
    if hue:
        huewrap = [hue]
        if hue == 'time':
            huetitle = 'Time (s)'
            starttime = section['time'].min()
            section['time'] = ((section['time']-starttime)/1000).apply(int)
        elif hue == 'currentobject':
            huetitle = 'Phase'
    
    section = section[features+huewrap].reset_index(drop=True)
    
    plot = sns.pairplot(data = section, diag_kind='hist', diag_kws={'bins': 20}, hue=hue)
    plt.suptitle('Subject {}, Trial {}, {}, {}'.format(subject, trial, phasestring, featurestring))
    
    plot._legend.set_title(huetitle)
        
    plt.show()

def trialFeatureAvgPairplot(data, statistic='mean', subjects='all', features='all', phases='all'):
    """A function to visualize a pairplot of a feature statistics and reaction times over a set of subject(s).

    Args:
        data (_type_): A dataframe of index: timestep, columns: features and rt, subject, trial, currentobject.
        statistic (str, optional): Descriptive statistic from: ('mean', 'median', 'max', 'min', 'std'). Defaults to 'mean'.
        subjects (int or list, optional): Subject(s) to include. Defaults to 'all'.
        features (str, optional): Feature(s) to include. Defaults to 'all'.
        phases (str, optional): Phase(s) to include. Defaults to 'all'.
    """
    
    section = data.copy()
    
    subjectstring = 'All Subjects'
    if subjects == 'all':
        pass
    else:
        if isinstance(subjects, int):
            subjects = [subjects]
        subjectstring = 'Subject(s): {}'.format(', '.join([str(s) for s in subjects]))
        section = section.query('subject in {}'.format(subjects))
    
    phasestring = 'All Phases'
    if phases == 'all':
        pass
    else:
        section = section.query('currentobject in "{}"'.format(phases))
        phasestring = 'Phase(s): {}'.format(phases)

    featurestring = 'All Features'    
    if features == 'all':
        features = ['right_gaze_x', 'right_gaze_y', 'left_gaze_x', 'left_gaze_y', 'right_pupil', 'left_pupil']
    else:
        if isinstance(features, str):
            features = [features]
        featurestring = 'Feature(s):'+', '.join(features)

    section = section[features+['subject', 'trial', 'rt']].reset_index(drop=True)
    if statistic == 'mean':
        section = section.groupby(['subject', 'trial']).mean()
    elif statistic == 'median':
        section = section.groupby(['subject', 'trial']).median()
    elif statistic == 'max':
        section = section.groupby(['subject', 'trial']).max()
    elif statistic == 'min':
        section = section.groupby(['subject', 'trial']).min()
    elif statistic == 'std':
        srt = section.groupby(['subject', 'trial']).mean()['rt']
        section = section.groupby(['subject', 'trial']).std()
        section['rt'] = srt
    
    sns.pairplot(data = section, diag_kind='hist', diag_kws={'bins': 20})
    plt.suptitle('Feature Averages Per Trial: {}, {}, {}'.format(subjectstring, phasestring, featurestring))
    
    plt.show()