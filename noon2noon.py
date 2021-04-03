import numpy as np 
import pandas as pd
import glob 

def main(path):
    all_data = pd.DataFrame()
    
    #looping through each file in path and appending to dataframe
    for file in glob.glob(path + '*'):
        data = pd.read_csv(file)
        all_data = pd.concat([all_data, data])
    all_data = all_data.sort_values('Timestamp')
    all_data = all_data.drop(columns = ['Unnamed: 0'])
    
    #getting rid of first 0h to 12h 
    all_data.index = pd.to_datetime(all_data['Timestamp'])
    all_data = all_data.drop(columns = ['Timestamp'])
    first_time = str(all_data.index[0])
    end_time = first_time[:11] + '11:59:59'
    first_12h = all_data.query("@first_time <= index <= @end_time")
    all_data = all_data.iloc[len(first_12h):] #this is cool

    #looping for all data
    while len(all_data) > 0:
        noon = str(all_data.index[0])
        next_day = str(int(noon[8:10]) + 1).zfill(2)
        noon_next_day = noon[:8] + next_day + ' 11:59:59'
        noon2noon = all_data.query("@first_time <= index <= @noon_next_day")
        all_data = all_data.iloc[len(noon2noon):]

        noon2noon.to_csv('noon2noon_' + noon[:10] + '.csv')

main('/Users/emiliolanzalaco/Documents/Smoking_Habit_Tracker/HR_CSV_Data/')

