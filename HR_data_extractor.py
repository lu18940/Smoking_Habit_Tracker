import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta, date

def main():
    start_date = date(2021, 3, 15)
    end_date = date(2021, 3, 16)
    HR_list, Timestamp_list = extract_HR(start_date, end_date)

    plt.plot(HR_list)
    plt.show()

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def extract_HR(start_date, end_date):
    Timestamp_list = np.empty([1,1])
    HR_list = np.empty([1,1])

    for curr_date in daterange(start_date, end_date):
        print(curr_date)
        if os.path.exists('Garmin Raw Data/{}'.format(curr_date)):
            for i in range(1,10):
                Garmin_data_filename = 'Garmin Raw Data/{}'.format(curr_date) +'/' + str(i) +'.csv'
                
                if os.path.exists(Garmin_data_filename):
                    Garmin_df = pd.read_csv(Garmin_data_filename)

                    HR_df = Garmin_df.loc[Garmin_df['Field 2']=='heart_rate']
                    temp_timestpamp_list = HR_df['Value 1'].to_numpy()
                    temp_HR_list = HR_df['Value 2'].to_numpy()

                    Timestamp_list = np.concatenate((Timestamp_list, temp_timestpamp_list), axis=None)
                    HR_list = np.concatenate((HR_list, temp_HR_list), axis=None)
                
                else:
                    pass

            index = 0
            remove_index = []
            for i in HR_list:
                if i < 20:
                    remove_index.append(index)
                index += 1

            HR_list = np.delete(HR_list, remove_index)
            Timestamp_list = np.delete(Timestamp_list, remove_index)

        else:
            pass

    return HR_list, Timestamp_list

if __name__ == '__main__':
    main()


    