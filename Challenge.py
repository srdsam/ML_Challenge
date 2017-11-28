import sys
import pandas as pd
import numpy as np
from ML_Model import regression, calc_activity2

# use Challenge.py < motion.tsv in bash after navigating to the correct directory.

dictionary = {'time': [pd.to_datetime('2017-05-23T05:45:16.325908+00:00')], 'walking': [0]}


def stdin_predict(line, model=regression):
    line = line.split("	")
    dictionary['time'].append(pd.to_datetime(str(line[0])))
    dictionary['walking'].append(int(line[2]))
    time_delta = pd.Timedelta(dictionary['time'][0] - dictionary['time'][-1])

    # ensuring data is only from a 48 hour period
    if time_delta < pd.Timedelta('-2 days'):
        del dictionary['time'][0]
        del dictionary['walking'][0]
        if time_delta < pd.Timedelta('-2 days'):
            del dictionary['time'][0]
            del dictionary['walking'][0]
            if time_delta < pd.Timedelta('-2 days'):
                del dictionary['time'][0]
                del dictionary['walking'][0]

    activity, activity_list, time_list = calc_activity2(dictionary)

    deviation = model.predict(float(activity.total_seconds()))
#    mean_sq_error = 2.02
#    variance_score = .38

    arange = np.linspace(deviation * 1.38, deviation * .62, 200, endpoint=True)

    if sum(arange) != 0:
        risk = (sum(12.5 < arange)/200) * 100
    else:
        risk = "Undefined"

    return risk, deviation, time_delta


for line in sys.stdin.readlines():
    glucose_predict, deviation, timedelta = stdin_predict(line, model=regression)
    print(line)
    print("Risk of Deviation above 12.5: " + str(glucose_predict) + "%")
    print("Predicted Deviation: " + str(float(deviation)))
    print(timedelta)


