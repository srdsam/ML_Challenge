import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def extract_data(tsv='motion.tsv', csv='blood-glucose.csv'):
    with open(str(tsv)) as txt:
        tsv_data = pd.read_csv(txt, sep="\t", names=["stationary", "walking", "running", "automotive", "cycling"],
                               parse_dates=True, infer_datetime_format=True)
        txt.close()
    with open(str(csv)) as cxv:
        csv_data = pd.read_csv(cxv, sep=",", names=["glucose"], parse_dates=True, infer_datetime_format=True)
        cxv.close()
    print("Data read\n")
    return tsv_data, csv_data


# calculates activity of series (not yet partitioned/truncated)
def calc_activity(panda_series):
    walking = panda_series
    activity = pd.Timedelta('0 days')
    activity_list = []

    for i in range(len(walking)):
        if i == 0:
            pass
        elif walking[i - 1] == 1 and walking[i] == 0:
            activity = activity + pd.to_datetime(walking.index[i]) - pd.to_datetime(walking.index[i - 1])
        elif walking[i - 1] == 1 and walking[i] == 1:
            activity = activity + pd.to_datetime(walking.index[i]) - pd.to_datetime(walking.index[i - 1])
        else:
            pass
        activity_list.append(activity.total_seconds())

    time_list = pd.to_datetime(tsv['walking'].index.values)

    return activity, activity_list, time_list


def calc_activity2(dictionary):
    walking = dictionary['walking']
    time__ = dictionary['time']
    activity = pd.Timedelta('0 days')

    activity_list = []

    for i in range(len(walking)):
        if i == 0:
            pass
        elif walking[i - 1] == 1 and walking[i] == 0:
            activity = activity + pd.to_datetime(time__[i]) - pd.to_datetime(time__[i - 1])
        elif walking[i - 1] == 1 and walking[i] == 1:
            activity = activity + pd.to_datetime(time__[i]) - pd.to_datetime(time__[i - 1])
        else:
            pass
        activity_list.append(activity.total_seconds())

    time_list = pd.to_datetime(time__)

    return activity, activity_list, time_list


# this calculates the axis values for time spent walking.
def calc_graph(panda_series):
    t = pd.date_range(min(panda_series.index.values), max(panda_series.index.values), freq='15min')
    truncated_fourteight = []
    activity_ = []

    for i in range(len(t)):
        fourtyeight = pd.date_range(t[i] - pd.Timedelta('2 days'), t[i], freq='15min')
        truncated_fourteight.append(panda_series.truncate(before=fourtyeight[0], after=fourtyeight[-1]))
    print("Data truncated\n")

    for i in range(len(truncated_fourteight)):
        activity, activity_list, time_list = calc_activity(truncated_fourteight[i])
        activity_.append(activity.total_seconds())
        print('|', end="")
    print("Activity calculated \n")

    return activity_, t


# calculates standard deviation of a series to return std and time
def standard_deviation(panda_series):
    panda_series = panda_series.interpolate(method='time')
    time = pd.date_range(min(panda_series.index.values), max(panda_series.index.values), freq='15min')
    truncated_fourteight = []
    standard_dev = []

    for i in range(len(panda_series)):
        fourtyeight = pd.date_range(time[i] - pd.Timedelta('2 days'), time[i], freq='15min')
        truncated_fourteight.append(panda_series.truncate(before=fourtyeight[0], after=fourtyeight[-1]))
    print("Blood Glucose Truncated\n")

    for i in range(len(truncated_fourteight)):
        standard_dev_i = np.std(np.asarray(truncated_fourteight[i]), keepdims=True)
        standard_dev.append(standard_dev_i)
    print("Standard Deviation of Truncated Values Calculated\n")

    return standard_dev, time


def create_frames(time, time_list, standard_dev, activity_2):
    time_series = list(time) + list(time_list)
    time_series = np.asarray(pd.Series(time_series).sort_values())

    glucose_frame = pd.DataFrame(standard_dev, time, columns=['glucose'])
    active_frame = pd.DataFrame(activity_2, time_list, columns=['walking'])

    frames = glucose_frame.combine_first(active_frame).dropna(thresh=1).interpolate(method='time')
    return frames, time_series


def train_model(frames):
    walking = frames['walking'].values[:, np.newaxis]
    glucose = frames['glucose']
    # training/test sets of data 70:30 training:test split for x data (walking - independent variable)
    X_train, X_test, y_train, y_test = train_test_split(walking, glucose, test_size=0.3, random_state=42)
    y_train = y_train.interpolate()
    y_test = y_test.interpolate()

    # linear model object
    regression = linear_model.LinearRegression()

    # fit data
    regression.fit(X_train, y_train)

    print("Model Trained")

    return regression


tsv, csv = extract_data()
panda_series2 = tsv['walking']
activity_2, time_list = calc_graph(panda_series2)
standard_dev, time = standard_deviation(csv['glucose'])
frames, time_series = create_frames(time, time_list, standard_dev, activity_2)
regression = train_model(frames)
