import os
import sys
import pandas as pd
from src.logger import logging
from src.utils import load_object
from src.exception import CustomException



class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        preprocessor = load_object(os.path.join('artifacts','preprocessor.pkl'))
        model = load_object(os.path.join('artifacts','model.pkl'))
        logging.info(features)
        processed_data = preprocessor.transform(features)
        preds = model.predict(processed_data)
        return preds





class CustomData:
    def __init__(self,
                 Airline:str,
                 Source:str,
                 Destination:str,
                 Journey_day:str,
                 Class:str,
                 Departure:str,
                 Arrival:str,
                 Total_stops:str,
                 Duration_in_hours:int,
                 Day_of_journey:str,
                 Month_of_journey:str
                 ):

        self.Airline = Airline,
        self.Source = Source,
        self.Destination = Destination,
        self.Journey_day = Journey_day,
        self.Class = Class,
        self.Departure = Departure,
        self.Arrival = Arrival,
        self.Total_stops = Total_stops,
        self.Duration_in_hours = Duration_in_hours,
        self.Day_of_journey = Day_of_journey,
        self.Month_of_jourey = Month_of_journey
        self.Date_of_journey = "2023" + str(self.Month_of_jourey) #+ str(self.Day_of_journey)

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Date_of_journey": int(self.Date_of_journey),
                "Journey_day": self.Journey_day,
                "Airline": self.Airline,
                "Class": self.Class,
                "Source":self.Source,
                "Departure": self.Departure,
                "Total_stops": self.Total_stops,
                "Arrival": self.Arrival,
                "Destination": self.Destination,
                "Duration_in_hours":self.Duration_in_hours,
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)        
