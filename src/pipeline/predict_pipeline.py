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
        model = load_object(os.path.join('artifacts','preprocessor.pkl'))
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
                 Days_Left:int,
                 Duration_in_hours:int,
                 Date_of_journey:int
                 ):

        self.Airline = Airline,
        self.Source = Source,
        self.Destination = Destination,
        self.Journey_day = Journey_day,
        self.Class = Class,
        self.Departure = Departure,
        self.Arrival = Arrival,
        self.Total_stops = Total_stops,
        self.Days_left = Days_Left,
        self.Duration_in_hours = Duration_in_hours,
        self.Date_of_journey = Date_of_journey

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Date_of_journey": [self.Date_of_journey],
                "Journey_day": [self.Journey_day],
                "Airline": [self.Airline],
                "Class": [self.Class],
                "Source":[self.Source],
                "Departure": [self.Departure],
                "Total_stops": [self.Total_stops],
                "Arrival": [self.Arrival],
                "Destination": [self.Destination],
                "Duration_in_hours":[self.Duration_in_hours],
                "Days_left":[self.Days_left],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)        