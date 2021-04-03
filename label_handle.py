import numpy as np 
import pandas as pd
import glob
import os


def main(label_path):
    for file in glob.glob(label_path):
        label_data = pd.read_csv(file)
        hr_files = sorted(glob.glob('noon2noon_data/*'))
        for day in hr_files:
            print(day[25:35])
            hr_data = pd.read_csv(day)
            hr_data['sleep time'] = int(0)
            time = label_data.loc[label_data['Day'] == day[25:35], 'Sleep time'].to_frame()
            for a in time.values:
                for b in a:
                    hr_data.loc[hr_data['Timestamp'].str.contains(str(b)), 'sleep time'] = int(1)
            hr_data.to_csv('noon2noon_data/noon2noon_' + day[25:35] + '.csv')
main('/Users/emiliolanzalaco/Documents/Smoking_Habit_Tracker/sleep_labels/sleep_labels.csv')

#df.loc[df[theme].isnull(), theme] = int(0)
#, ['sleep time']] = int(1)