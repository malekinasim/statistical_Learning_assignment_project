from utils.DataLoader import DataLoader
from feature_engineering.politic.PartyFeatures import PoliticPartyFeatures
import pandas as pd

class PoliticDataLoader(DataLoader):

    @staticmethod
    def __expand_rows(df):
        new_rows = []
        for row in df.to_dict(orient='records'):
            base_year = int(row['Election Year'])
            for add in range(1, 4):
                if (base_year + add) > 2023:
                    break
                new_row = row.copy()
                new_row['Election Year'] = base_year + add
                new_rows.append(new_row)
        
        new_rows_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([df, new_rows_df], ignore_index=True)
        combined_df = combined_df.sort_values(by=['Municipality', 'Election Year']).reset_index(drop=True)
        return combined_df

    @staticmethod
    def load_political_data_main(file_path):
        data = DataLoader.load_data(file_path)
        data['Municipality'].ffill(inplace=True)

        # Split Municipality into Code and Name
        data[['Code', 'Municipality']] = data['Municipality'].str.split(' ', n=1, expand=True)

        # Expand years
        data = PoliticDataLoader.__expand_rows(data)

        # Extract party columns by excluding known columns
        # remove Other Parties column for removing Correlation between parties because increasing one party will decrease the other parties
        # and this will cause the correlation between them invalid regression Model
        known_cols = {'Code', 'Municipality', 'Election Year'}
        party_columns = [col for col in data.columns if col not in known_cols]

        political_party_features_list = []

        for row in data.to_dict(orient='records'):
            municipality = row['Municipality']
            election_year = row['Election Year']
            total_seats_count = sum(row[party] for party in party_columns)
            
            for party_name in party_columns:
                seats_count = row[party_name]
                party = PoliticPartyFeatures(election_year, municipality, party_name, seats_count, total_seats_count)
                political_party_features_list.append(party)

        df = DataLoader.to_dataFrame(political_party_features_list)

        # Pivot
        df_wide = df.pivot(index=['Municipality', 'Year'], columns='Name', values='Seats_percentage').reset_index()

        left_parties = ['The Green Party', 'The Social Democratic Party', 'The Left Party']
        right_parties = ['The Moderate Party', 'The Christian Democratic Party', 'The Sweden Democrats']

        
        df_wide['Left_share'] = df_wide[left_parties].sum(axis=1)
        df_wide['Right_share'] = df_wide[right_parties].sum(axis=1)


        df_wide['Left_minus_Right'] = df_wide['Left_share'] - df_wide['Right_share']

        # Flatten MultiIndex columns
        df_wide.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df_wide.columns]

        return df_wide[['Municipality', 'Year', 'Left_share', 'Right_share', 'Left_minus_Right']]

