import os
import pandas as pd
from constants import STORAGE_DIR, FREQUENCY, FEATURE_NAMES, WEEK_MEASUREMENTS, DOWNSAMPLE_VALUE
from sklearn.preprocessing import MinMaxScaler

def feature_engr(filename):
    df = pd.read_csv(filename)
    df = df.iloc[::DOWNSAMPLE_VALUE, :]
    # change in SOC
    df['delta_soc'] = df['SOC'].diff().fillna(0)

    # df['temp'] = df['temp']
    # change in mileage
    df['delta_mile'] = df['Mileage'].diff().fillna(0)
    df.loc[df['delta_mile'] < 0, 'delta_mile'] = 0 # mileage cannot decrease

    # fix home charging status: when mileage is changing or not charging, user is not at home charging
    df.loc[df['delta_mile'] > 0, 'Home'] = 0
    df.loc[df['delta_soc'] <= 0, 'Home'] = 0

    # week number
    df['week_num'] = df.reset_index(drop=True).index//WEEK_MEASUREMENTS

    # weekly mileage
    df['weekly_mile'] = df.groupby('week_num')['delta_mile'].cumsum()

    # get charging status: 0 = not charging (when delta_soc is negative), 1 = charging (when delta_soc is positive)
    df['charging_status'] = df['delta_soc'].apply(lambda x: 1 if x>0 else 0)

    # calculate cycle number where 0.5 cycles = only charging || only discharging
    # detect cycle_change = 0 --> 1 discharge-charge cycle
    # cumulative sum = number of cycles
    df['cycle_diff'] = df['charging_status'].diff().fillna(0).abs()
    df['cycle_num'] = df['cycle_diff'].cumsum()/2
    

    # add cycle change threshold, where SOC% change < threshold does not count as a cycle
    df['cycle_thres'] = df.groupby('cycle_num')['SOC'].transform(lambda x: 0 if ((x.max()-x.min())<2) else 1)
    df['cycle_change'] = df['cycle_diff'].copy()
    df.loc[df['cycle_thres'] == 0, 'cycle_change'] = 0
    df['cycle_num'] = df['cycle_change'].cumsum()/2

    # redefine charging status to reflect charging threshold
    # when max SOC comes before min SOC --> discharging
    # when min SOC comes before max SOC --> charging
    df['charging_status'] = df.groupby('cycle_num')['SOC'].transform(lambda x: 1 if ((x.idxmax()-x.idxmin())>0) else 0)

    # redefine cycle number to reflect new charging statuses
    df['cycle_diff'] = df['charging_status'].diff().fillna(0).abs()
    df['cycle_num'] = df['cycle_diff'].cumsum()/2
    # cycle depth = upper cutoff SOC% - lower cutoff SOC%
    df['depth'] = df.groupby('cycle_num')['SOC'].transform(lambda x: (x.max()-x.min()))
    # depth of discharge = cycle depth only during discharging
    df['dod'] = df['depth'].copy()
    df.loc[df['charging_status']==1, 'dod'] = 0
    # charging power level (1=slow, 2=fast, 3=rapid)
    # 1 < 0.054%/min, 2 < 0.417%/min, 3 >= 0.417%/min
    df['charging_power_level'] = df.groupby('cycle_num')['delta_soc'].transform(lambda x: 0 if (x.max()<=0)
                                                                                else (1 if (x.max()<0.054*int(FREQUENCY[:-3]))
                                                                                else (2 if (x.max()<0.417*int(FREQUENCY[:-3])) else 3)))
    df.loc[df['charging_status']==0, 'charging_power_level'] = 0
    # calculate charging energy from change in SOC and usable capacity (83.6 kwh)
    df['charging_energy_kwh'] = df['SOC'].copy()
    df.loc[df['charging_status']==0, 'charging_energy_kwh'] = 0
    df['charging_energy_kwh'] = df.groupby('cycle_num')['charging_energy_kwh'].transform(lambda x: (x.max()-x.min())/100*83.6)
    # weekly cycle number
    df['weekly_cycle'] = df.groupby('week_num')['cycle_diff'].cumsum()/2 
    df.drop(columns=[
                    'cycle_change',
                    'cycle_diff',
                    'cycle_thres',
                    'depth'
                    ], inplace=True)

    df['velocity'] = df['delta_mile'].apply(lambda x: x*60/DOWNSAMPLE_VALUE)
    df['delta_energy'] = df["delta_soc"].apply(lambda x: x * 83.6/85)
    # df['delta_temp'] = df['temp'].diff().fillna(0)
    if df[df.isna().any(axis=1)].shape[0] != 0 or df.shape[0] == 0:
        print (f"Skipping! Found some Nans OR \"Empty dataframe\" in the features of {filename}")
        success = 0

    else:
        backslash = '\\'
        df.to_csv(f"{STORAGE_DIR}/{FREQUENCY}/{filename.split(backslash)[-1].split('/')[-1]}", index=False)
        success = 1

    return df, success


def feature_engr_sync(filename, scaler) -> pd.DataFrame:
    df, success = feature_engr(filename)
    if success:
        scaler.partial_fit(df.loc[:,FEATURE_NAMES]) # fit scaler for normalization on relevant features
    return df

def feature_engr_parallel(filename):
    [df, success] = feature_engr(filename)
    return success

