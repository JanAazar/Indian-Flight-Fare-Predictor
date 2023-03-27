import sys
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor

from src.utils import evaluate_models
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    model_obj_filepath = os.path.join("artifacts","model.pkl")


class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def ModelTrainer(self,train_arr,test_arr):
        try:
            models = {"RandomForestRegressor":RandomForestRegressor()}
            
            params = {            
                    "RandomForestRegressor":{
                        'n_estimators': [64,128,256],
                        'max_depth': [15, 20],
                        "min_samples_split": [10, 15],
                        "min_samples_leaf": [4, 8]
                    }              
                }

            logging.info("Defined Model and Parameter Grids.")

            train_x = train_arr[:,:-1]
            train_y = train_arr[:,-1].toarray().reshape(-1,1)
            test_x = test_arr[:,:-1]
            test_y = test_arr[:,-1].toarray().reshape(-1,1)

            logging.info("Created X and Y arrays")


            model_report, best_parameters = evaluate_models(models,params,train_x,train_y,test_x,test_y)
            best_score = max(list(model_report.values()))

            if best_score<0.6:
                raise CustomException("No Good Model Found")
            
            best_score_index = list(model_report.values()).index(best_score)
            best_model_name = list(model_report.keys())[best_score_index]
            best_params = best_parameters[best_score_index]
            best_model = models[best_model_name].set_params(**best_params)
            

            save_object(
                file_path=self.model_trainer_config.model_obj_filepath,
                obj=best_model
            )

            logging.info("Saved Best Model")
            logging.info(best_score)

            return best_model_name,best_score,best_params
    
        except Exception as e:
            raise CustomException(e,sys)
    

        































