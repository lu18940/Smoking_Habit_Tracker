import numpy as np 
import pandas as pd
import glob

def main(path):
    for file in glob.glob(path):
        data = pd.read_csv(file)
        HR_files = glob.glob('noon2noon_data/*')
        print(data.loc(data['Day']))

main('/Users/emiliolanzalaco/Documents/Smoking_Habit_Tracker/sleep_labels/sleep_labels.csv')