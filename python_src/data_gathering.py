
import pandas as pd
from sodapy import Socrata
from sklearn.preprocessing import LabelEncoder

socrata_domain = "data.cdc.gov"
socrata_dataset_identifier = "n8mc-b4w4"

MyAppToken = "SQV4rZQAtpCh6MWLy2zXrZhsE"
username = "gezi12zai@gmail.com"
password = "Gezi12zao@Gezi12zai"
timeout_seconds = 120
# Example authenticated client (needed for non-public datasets):
client = Socrata("data.cdc.gov", MyAppToken, username=username, password=password, timeout=timeout_seconds)

query_negative = """
select 
    case_month, 
    res_state,
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
    sex in('Female', 'MALE', 'Other')
    and race in('American Indian/Alaska Native', 'Asian', 'Black', 'Multiple/Other', 'Native Hawaiian/Other Pacific Islander', 'White')
    and death_yn in('No')
limit
    100000
"""

query_positive = """
select 
    case_month, 
    res_state,
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
    sex in('Female', 'MALE', 'Other')
    and race in('American Indian/Alaska Native', 'Asian', 'Black', 'Multiple/Other', 'Native Hawaiian/Other Pacific Islander', 'White')
    and death_yn in('Yes')
limit
    100000
"""
# New column order

new_order = ['case_month', 'res_state', 'age_group', 'sex', 'race', 'process', 'exposure_yn', 'symptom_status',
             'hosp_yn', 'icu_yn', 'underlying_conditions_yn', 'death_yn']

results_negative = client.get(socrata_dataset_identifier, query=query_negative)
print('negative finished')
results_positive = client.get(socrata_dataset_identifier, query=query_positive)
print('positive finished')

result = results_negative + results_positive

results_df = pd.DataFrame.from_records(result)

# Reorder columns
results_df = results_df[new_order]

# fill missing values
results_df['underlying_conditions_yn'] = results_df['underlying_conditions_yn'].fillna('Unknown')

# convert the label values
results_df['death_yn'] = results_df['death_yn'].replace({'No': 0, 'Yes': 1})
# convert string attributes to numerical values
converted_parts = ['case_month', 'res_state', 'age_group', 'sex', 'process', 'exposure_yn', 'symptom_status',
             'hosp_yn', 'icu_yn', 'underlying_conditions_yn', 'death_yn']
results_df[converted_parts] = results_df[converted_parts].apply(LabelEncoder().fit_transform)
results_df['race'] = results_df['race'].replace({
    'American Indian/Alaska Native': 'American_Indian_Alaska_Native',
    'Multiple/Other': 'Multiple_Other'
})

# save the data
results_df.to_csv('../data/cdc_data.csv', index=False)