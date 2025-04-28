class CityStatus:
    def __init__(self, year, municipality, economic_features, education_features, demographic_features, politic_features):
        self.__year = year
        self.__municipality = municipality
        self.__economic_features = economic_features
        self.__education_features = education_features
        self.__demographic_features = demographic_features
        self.__politic_features = politic_features
    def get_year(self):
        return self.__year  
    def get_municipality(self):
        return self.__municipality  
    def get_economic_features(self):    
        return self.__economic_features
    def get_education_features(self):   
        return self.__education_features
    def get_demographic_features(self):
        return self.__demographic_features          
    def get_politic_features(self):
        return self.__politic_features  
