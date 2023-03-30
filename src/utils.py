import sys
import os

import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score



def save_object(file_path,obj):
    '''This funcion saves an object in a specified loaction.'''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:            
            dill.dump(obj,file)
    except Exception as e:
        raise(CustomException(e,sys))
            

def date_to_int(val):
    try:
        val = val.replace("-","")
        val = int(val)
        return (val)
    except Exception as e:
        raise(CustomException(e,sys))
                 
def evaluate_models(models,params,train_x,train_y,test_x,test_y):

    try:

        r2_report = {}
        parameters_report = []
        best_weights = []
        
        for model_name,model in models.items():
            model_params = params[model_name]


            logging.info("Defined Model Parameters")    
            search = RandomizedSearchCV(estimator=model,param_distributions=model_params,cv=3)
            logging.info("Intitited Search")
            search.fit(train_x,train_y)
            logging.info("fitted Search")
            best_model = search.best_estimator_
            logging.info("Found Best Model")

            best_model.fit(train_x,train_y)
            logging.info("Fitted the best model on training data")
            y_pred = best_model.predict(test_x)

            r2_scored = r2_score(test_y,y_pred)

            parameters_report.append(search.best_params_)
            best_weights.append(best_model.feature_importances_)
            r2_report[model_name]  = r2_scored

        return r2_report,parameters_report,best_weights
    
    except Exception as e:
        raise(CustomException(e,sys))
        

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)    




        
        
