from utils.DataLoader import DataLoader
from feature_engineering.politic.PartyFeatures import PoliticPartyFeatures
class PoliticDataLoader(DataLoader):
    @staticmethod
    def load_political_data(file_path):
        """Load political data and return a list of PoliticPartyFeatures objects."""
        data = DataLoader.load_data(file_path)
        political_party_features_list = []
        rows=data.itertuples()

        rows = data.to_dict(orient='records')
        i=0
        while i<len(data):
            row=rows[i]
            municipality = row['Municipality']
            total_seats_count=0
            
            k=0
            while k<3:
                row=rows[i+k]
                election_year =row['Election Year'] 
                total_seats_count=sum(row[party_name] for party_name in data.columns[2:])
                for party_name in data.columns[2:]:
                    seats_count = row[party_name] 
                    j=0
                    while j<4:
                        party = PoliticPartyFeatures(election_year+j,municipality,party_name, seats_count, total_seats_count)
                        political_party_features_list.append(party)
                        j+=1
                k+=1
            i+=3
        return political_party_features_list