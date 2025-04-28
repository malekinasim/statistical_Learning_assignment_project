class PoliticPartyFeatures:
    def __init__(self, year,municipality,name, seats_count,total_seats_count):
        self.__year=year
        self.__municipality=municipality
        self.__name = name 
        self.__seats_count = seats_count 
        self.__seats_percentage =self.__calculate_percentage(total_seats_count)
    
    def __calculate_percentage(self, total_seats):
        return round(self.__seats_count / total_seats,2) * 100
    def get_name(self):
        return self.__name
    def get_seats_count(self):
        return self.__seats_count 
    def get_seats_percentage(self):
        return self.__seats_percentage  
    def get_year(self):
        return self.__year
    def get_municipality(self):
        return self.__municipality
    def __str__(self):  
        return f"Year: {self.__year}, Municipality: {self.__municipality}, Party: {self.__name}, Seats Count: {self.__seats_count}, Seats Percentage: {self.__seats_percentage}"