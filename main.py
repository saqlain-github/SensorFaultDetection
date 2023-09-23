import os
import sys
import datetime
from io import BytesIO
import pandas as pd
import numpy as np
import json
from sklearn.metrics import classification_report
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import TARGET_COLUMN, SAVED_MODEL_DIR, SCHEMA_FILE_PATH
from fastapi import FastAPI, UploadFile, File
from sensor.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response, JSONResponse
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware

env_file_path=os.path.join(os.getcwd(),"env.yaml")

def set_env_variable(env_file_path):

    if os.getenv('MONGO_DB_URL',None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL']=env_config['MONGO_DB_URL']

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/sensorfaultdetection/health")
async def root():
    try:
        return JSONResponse(
            status_code=200,
            content = {
                "message":"OK",
                "success": True,
                "response":"Application Running Successfully!!"
                }
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content = {
                "message":"Fail",
                "success": False,
                "response":"Error while starting the Application!!"
                }
        )

@app.get("/sensorfaultdetection/train")
async def train_route():
    try:
        set_env_variable(env_file_path)
        train_pipeline = TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        train_pipeline.run_pipeline()
        return JSONResponse(
            status_code=200,
            content = {
                "message":"OK",
                "success": True,
                "response":"Training Pipeline successful !!"
                }
        )
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content = {
                "message":"Fail",
                "success": False,
                "response":f"Training Pipeline failed !! {e}"
                }
        )

@app.post("/sensorfaultdetection/predict")
async def predict_route(file: UploadFile):
    try:
        #get data from user csv file
        #conver csv file to dataframe
        df = pd.DataFrame()
        if file.filename.endswith(".csv"):
            try:
                # Read the CSV file and convert it into a DataFrame
                csv_content = await file.read()
                csv_string = csv_content.decode("utf-8")
                csv_file = BytesIO(csv_content)
                df = pd.read_csv(csv_file)

                # You now have a DataFrame 'df' that contains the CSV data

            except Exception as e:
                return {"success": False, "message": f"Error: {str(e)}"}

        # df = None
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available")

        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        
        input_feature_df = df.drop(columns=[TARGET_COLUMN], axis=1)
        input_feature_df = input_feature_df.replace("na", np.nan)
        input_feature_df = input_feature_df.drop(read_yaml_file(SCHEMA_FILE_PATH)["drop_columns"],axis=1)
        
        
        target_feature_train_df = df[TARGET_COLUMN]
        target_feature_train_df = target_feature_train_df.replace( TargetValueMapping().to_dict())
        
        y_pred = model.predict(input_feature_df)
        df['predicted_column'] = y_pred
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
        
        report = classification_report(target_feature_train_df, y_pred, output_dict=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_str = f"prediction/{timestamp}.csv"
        df.to_csv(out_str)

        # Return the JSON response containing the classification report
        return JSONResponse(
            status_code=200,
            content = {
                "message":"OK",
                "success": True,
                "response":"Prediction Pipeline successful !!",
                "classfication_report": report,
                "output_file" : out_str
                }
        )
        #return json.loads(df[["class","predicted_column"]].to_json(orient="records"))
    except Exception as e:
        raise JSONResponse(
            status_code=502,
            content = {
                "message":"Fail",
                "success": False,
                "response":"Prediction Pipeline failed !!"
                }
        )

def main():
    try:
        set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__=="__main__":
    main()
    # app_run(app, host=APP_HOST, port=APP_PORT)
