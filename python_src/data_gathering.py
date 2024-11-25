
import pandas as pd
from sodapy import Socrata
from sklearn.preprocessing import LabelEncoder

socrata_domain = "data.cdc.gov"
socrata_dataset_identifier = "n8mc-b4w4"

MyAppToken = "SQV4rZQAtpCh6MWLy2zXrZhsE"
username = "gezi12zai@gmail.com"
password = "Gezi12zao@Gezi12zai"
timeout_seconds = 600
# Example authenticated client (needed for non-public datasets):
client = Socrata("data.cdc.gov", MyAppToken, username=username, password=password, timeout=timeout_seconds)
sample_number = 20000
races = ['White', 'Black', 'Asian']
labels = ['Yes', 'No']
Sexs= ['Male', 'Female']

result = []
for race in races:
    for label in labels:
        for sex in Sexs:
            query = """
            select 
                age_group,
                sex,
                race,
                process,
                exposure_yn,
                symptom_status,
                hosp_yn,
                icu_yn,
                underlying_conditions_yn,
                death_yn
            where
                sex in('{}')
                and race in('{}')
                and death_yn in('{}')
            limit
                {}
            """.format(sex, race, label, sample_number)

            result += client.get(socrata_dataset_identifier, query=query)
            print("finish {}_{}_{} query".format(race, sex, label))

# New column order

new_order = ['race', 'sex', 'age_group', 'process', 'exposure_yn', 'symptom_status',
             'hosp_yn', 'icu_yn', 'underlying_conditions_yn', 'death_yn']

results_df = pd.DataFrame.from_records(result)

# Reorder columns
results_df = results_df[new_order]

# fill missing values
results_df['underlying_conditions_yn'] = results_df['underlying_conditions_yn'].fillna('Unknown')

# Set 'race' and 'sex' as the MultiIndex
results_df = results_df.set_index(['race', 'sex'])

# save the data
results_df.to_csv('../data/cdc_data.csv')