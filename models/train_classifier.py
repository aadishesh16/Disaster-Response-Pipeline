
import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    '''
    INPUT 
        database_filepath - Filepath used for importing the database     
    OUTPUT
        Returns the following variables:
        X - Returns the input features.  Specifically, this is returning the messages column from the dataset
        Y - Returns the categories of the dataset.  This will be used for classification based off of the input X
        y.keys - Just returning the columns of the Y columns
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    conn = engine.connect()
    df = pd.read_sql("Select * from messages_tables",conn)
    X = df.message
    Y = df.iloc[:, 4:]
    
    return X, Y, Y.keys()

def tokenize(text):
    '''
    INPUT 
        text: Text to be processed   
    OUTPUT
        Returns a processed text variable that was tokenized, lower cased, stripped, and lemmatized
    '''
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    final_tokens =[]
    for tok in tokens:
        final_tok = lemmatizer.lemmatize(tok).lower().strip()
        final_tokens.append(final_tok)
    return final_tokens

def build_model(X_train,y_train):
    '''
    INPUT 
        X_Train: Training features for use by GridSearchCV
        y_train: Training labels for use by GridSearchCV
    OUTPUT
        Returns a pipeline model that has gone through tokenization, count vectorization, 
        TFIDTransofmration and created into a ML model
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {  
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__criterion': ['entropy', 'gini']
    }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT 
        pipeline: The model that is to be evaluated
        X_test: Input features, testing set
        y_test: Label features, testing set
        category_names: List of the categories 
    OUTPUT
        This method does nto specifically return any data to its calling method.
        However, it prints out the precision, recall and f1-score
    '''
    # predict on test data
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values,y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:,i])))

def save_model(model, model_filepath):
    '''
    Saves the model to disk
    INPUT 
        model: The model to be saved
        model_filepath: Filepath for where the model is to be saved
    OUTPUT
        While there is no specific item that is returned to its calling method, this method will save the model as a pickle file.
    '''    
    with open(model_filepath,"wb") as f:
        pickle.dump(model,f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train,Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        
        ###WILL NEED TO CXLEAN THIS UP
        print('TYPE OF MODEL')
        print(type(model))
        
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()