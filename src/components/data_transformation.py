import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import date_to_int
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_processor(self):
        try:
            cat_columns=['Journey_day','Airline','Source','Destination','Class','Departure','Arrival','Total_stops'
                         ]
            num_columns = ['Duration_in_hours','Date_of_journey']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ])
            

            logging.info("Created Categorical and Numerical Pipelines.")

            

            preprocesor = ColumnTransformer(
            [("cat_encoder",cat_pipeline,cat_columns),
             ("num_encoder",num_pipeline,num_columns)
             ]
            ) 

            logging.info("Created preprocessor")

            return preprocesor
        except Exception as e:
            raise(CustomException(e,sys))
        

    def Start_Data_Transformation(self,train_path,test_path):
        try:

            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            train_df['Date_of_journey'] = train_df['Date_of_journey'].apply(date_to_int)
            test_df['Date_of_journey'] = test_df['Date_of_journey'].apply(date_to_int)
            
            train_x_df = train_df.drop("Fare",axis=1)
            train_y_arr = train_df["Fare"].values
            test_x_df = test_df.drop("Fare",axis=1)
            test_y_arr = test_df["Fare"].values

            preprocesor = self.get_processor()

            train_x_arr = preprocesor.fit_transform(train_x_df)
            test_x_arr =  preprocesor.transform(test_x_df)

            logging.info("Applied transformations on train and test sets.")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_filepath,
                obj = preprocesor
            )
            logging.info("Saved Preprocessor")

            return (
                train_x_arr,
                train_y_arr,
                test_x_arr,
                test_y_arr,
                self.data_transformation_config.preprocessor_obj_filepath
            )

        except Exception as e:
            
            raise(CustomException(e,sys))







