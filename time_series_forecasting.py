import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv('noon2noon_data/noon2noon_2021-02-01.csv')

    # timestamps = hr_df['Timestamp'].to_list()
    date_time = pd.to_datetime(df.pop('Timestamp'), format='%Y.%m.%d %H:%M:%S')
    timestamp_s = date_time.map(datetime.datetime.timestamp)

    day = 24*60*60

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))

    # plt.plot(np.array(df['Day sin']))
    # plt.plot(np.array(df['Day cos']))
    # plt.xlabel('Time [h]')
    # plt.title('Time of day signal')
    # plt.show()

    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    plt.show()

if __name__ == '__main__':
    main()