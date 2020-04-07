# Disaster Response Pipeline Project
A Machine Learning bases web app to classify messages people posted during a disaster. I applied data engineering, natural language processing and random forest machine learning algorithm in the project. For front end, Bootstrap is used and Flask is used for the back end. 

## Dependencies:
1. Following packages are required for NLP:
- punkt
- wordnet
- stopwords
2. Python 3.5+
3. SQLlite Database Library:SQLalchemy
4. Python modules: Pandas, Numpy, Sci-kit learn,Plotly
5. Web App: Flask

## File descriptions:
It contains three main folders:

1. App:-
- **run.py** : Flask file to run the web app
- **templates/** : Contains two .html files.

2. Data:-
- **disaster_categories.csv** : Contains all the categories
- **disaster_messages.csv** : Contains all the messages sent during disaster
- **DisasterResponse.db** : Database containing messages and corresponding category values
- **process_data.py** : Python file for processing the two csv files and storing it into the database.

3. Models:
- **train_classify.py** : Machine Learning pipline to train, evaluate and export the data.
- **classifier.pkl** : Output of the machine learning pipeline stored in this file.

## Instructions:
Run the following commands in the project's root directory to set up your database and model.
- To run ETL pipeline that cleans data and stores in database `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
Run the following command in the app's directory to run your web app. `python run.py`

## Acknowledgements:
Udacity for the started code and FigureEight for the dataset.


