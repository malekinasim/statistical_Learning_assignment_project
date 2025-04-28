from feature_engineering.politic.PoliticDataLoader import PoliticDataLoader

def main():
    file_path = 'data/Dalarna_MunicipalElectionResults.xlsx'  
    political_data = PoliticDataLoader.load_political_data(file_path)

    for data in political_data:
        print(data) 
main()       
