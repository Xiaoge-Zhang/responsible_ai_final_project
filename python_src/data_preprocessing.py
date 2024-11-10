import pandas as pd
import random
pd.set_option('future.no_silent_downcasting', True)

# read the original data
raw_data = pd.read_csv('../data/latestdata.csv', low_memory=False)

# get rid of irrelavent data
irrelevent_cols = ['latitude','longitude','geo_resolution','date_onset_symptoms','date_admission_hospital',
                   'date_confirmation', 'travel_history_dates','travel_history_location','reported_market_exposure',
                   'additional_information','chronic_disease','source','sequence_available','date_death_or_discharge',
                   'notes_for_discussion','location','admin3','admin2','admin1','country_new','admin_id',
                   'data_moderator_initials', 'city', 'province', 'lives_in_Wuhan']

modified_data = raw_data.drop(columns=irrelevent_cols)

# drop cases with no outcomes
dropping_columns = ['outcome']
modified_data = modified_data.dropna(subset=dropping_columns)

# deal with missing and age area for age data
# first handle the age data that is like this "20-77"
for index, row in modified_data.iterrows():
    row_age = row['age']
    if isinstance(row_age, str) and '-' in row_age:
        try:
            lower, upper = map(float, row_age.split('-'))
            modified_data.at[index, 'age'] = (lower + upper) / 2
        except ValueError:
            parts = row_age.split('-')
            modified_data.at[index, 'age'] = float(parts[0])

# now for the missing values, we replace them with mode imputation
modified_data['age'] = pd.to_numeric(modified_data['age'], errors='coerce')
modified_data['age'] = modified_data['age'].fillna(modified_data['age'].mode().iloc[0])

# for missing values in sex, we replace it with the word "unknown"
modified_data['sex'] = modified_data['sex'].fillna('unknown')

# for missing values in symptoms and country, we do the same thing
modified_data['country'] = modified_data['country'].fillna('unknown')
modified_data['symptoms'] = modified_data['symptoms'].fillna('unknown')

# for missing values in column "travel_history_binary",  we set them to false
modified_data['travel_history_binary'] = modified_data['travel_history_binary'].fillna(False)\

# for the outcome, we first drop some undesirabel values
undesirable_values = ['Critical condition', 'critical condition, intubated as of 14.02.2020', 'Hospitalized',
                      'http://www.mspbs.gov.py/covid-19.php', 'Migrated', 'Migrated_Other', 'Receiving Treatment',
                      'not hospitalized', 'severe', 'severe illness',
                      'Symptoms only improved with cough. Currently hospitalized for follow-up',
                      'treated in an intensive care unit (14.02.2020)', 'Under treatment', 'unstable']
modified_data = modified_data[~modified_data['outcome'].isin(undesirable_values)]

# then, we convert rest of the rows to 1 or 0 depending on the outcome
values_to_convert_to_1 = ['dead', 'death', 'Deceased', 'died']
modified_data['outcome'] = modified_data['outcome'].isin(values_to_convert_to_1).astype(int)

modified_data.to_csv('../data/processed_data.csv', index=False)