import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta, date

def main():
    start_date = date(2021, 2, 1)
    end_date = date(2021, 2, 8)
    extract_HR(start_date, end_date)

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def extract_HR(start_date, end_date):
    UTC_REFERENCE = 631065600

    for curr_date in daterange(start_date, end_date):
        Timestamp_list = np.empty([1,1])
        HR_list = np.empty([1,1])

        if os.path.exists('Garmin Raw Data/{}'.format(curr_date)):
            for i in range(1,15):
                Garmin_data_filename = 'Garmin Raw Data/{}'.format(curr_date) +'/' + str(i) +'.csv'
                
                if os.path.exists(Garmin_data_filename):
                    Garmin_df = pd.read_csv(Garmin_data_filename)

                    HR_df = Garmin_df.loc[Garmin_df['Field 2']=='heart_rate']
                    Timestamp_df = Garmin_df.loc[Garmin_df['Field 1']=='timestamp']

                    current_timestamp = 0

                    for index, row in HR_df.iterrows():
                        for time_index, time_row in Timestamp_df.iterrows():
                            if time_index < index:
                                current_timestamp = int(time_row['Value 1'])
                            
                            else:
                                break

                        current_timestamp_16 = int(row['Value 1'])

                        ts_value = int(current_timestamp/2**16) * 2**16 + current_timestamp_16
                        real_time = datetime.datetime.utcfromtimestamp(UTC_REFERENCE + ts_value)
                        HR_df.loc[index, 'Value 1'] = real_time

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

            output_filename = 'HR CSV Data/' + str(curr_date) + '.csv'
            concatenated_data = {'Timestamp': Timestamp_list, 'Heart rate': HR_list}
            Output_df = pd.DataFrame(concatenated_data)
            Output_df.to_csv(output_filename)

        else:
            pass

if __name__ == '__main__':
    main()


    