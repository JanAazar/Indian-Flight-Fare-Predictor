from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Journey_day = request.form['Journey_day'],
            # Can you do the same with rows below 
            Airline = request.form['Airline'],
            Source = request.form['Source'],
            Destination = request.form['Destination'],
            Class = request.form['Class'],
            Departure = request.form['Departure'],
            Arrival = request.form['Arrival'],
            Total_stops = request.form['Total_stops'],
            Duration_in_hours = float(request.form['Duration_in_hours']),
            Month_of_journey = request.form['Month'],
            Day_of_journey = request.form['Day']
        )
    
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    #app.run(host="0.0.0.0")
    app.run(debug=True)

