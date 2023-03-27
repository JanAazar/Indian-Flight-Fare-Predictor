import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraining

@dataclass
class data_ingestion_config:
    train_data_path = os.path.join("datasets","train.csv")
    test_data_path = os.path.join("datasets","test.csv")
    raw_data_path = os.path.join("datasets","raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=data_ingestion_config()
    def start_data_ingestion(self):        
        logging.info("Starting the process of Data Ingestion")
        try:
            df=pd.read_csv(r"D:\ML_Project\Notebooks\flights.csv")
            df=df.drop("Flight_code",axis=1)
            df.drop_duplicates(inplace=True)
            df=df.sample(2000)
            logging.info("Read the data as a dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set,test_set = train_test_split(df,train_size=0.9,random_state=45)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Splited the data and saved them into their directories")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
                    



if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.start_data_ingestion()
    transformer = DataTransformation()
    train_arr,test_arr,_ = transformer.Start_Data_Transformation(train_path,test_path)
    trainer = ModelTraining()
    model_report = trainer.ModelTrainer(train_arr,test_arr)














