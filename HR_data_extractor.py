import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Timestamp_list = np.empty([1,1])
HR_list = np.empty([1,1])

for i in range(1,5):
    Garmin_data_filename = 'Garmin Raw Data/'+ str(i) +'.csv'
    Garmin_df = pd.read_csv(Garmin_data_filename)

    HR_df = Garmin_df.loc[Garmin_df['Field 2']=='heart_rate']
    temp_timestpamp_list = HR_df['Value 1'].to_numpy()
    temp_HR_list = HR_df['Value 2'].to_numpy()

    Timestamp_list = np.concatenate((Timestamp_list, temp_timestpamp_list), axis=None)
    HR_list = np.concatenate((HR_list, temp_HR_list), axis=None)

index = 0
remove_index = []
for i in HR_list:
    if i < 20:
        remove_index.append(index)
    index += 1

HR_list = np.delete(HR_list, remove_index)

plt.plot(HR_list)
plt.show()

    