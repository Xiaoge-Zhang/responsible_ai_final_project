import pandas as pd
import pickle
pd.set_option('future.no_silent_downcasting', True)

# read the original data
raw_data = pd.read_csv('../data/cdc_data.csv', low_memory=False)
races = ['American_Indian_Alaska_Native', 'Asian', 'Black', 'Multiple_Other', 'White']
for race in races:
    df = raw_data[raw_data['race'] == race]
    df = df.drop('race', axis=1)
    df.to_csv('../data/cdc_data_{}.csv'.format(race.replace(" ", "_")), index=False)


