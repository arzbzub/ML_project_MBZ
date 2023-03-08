from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from lightgbm import LGBMClassifier
from nltk.corpus import stopwords
import pandas as pd
import re
import spacy
nlp = spacy.load("es_core_news_lg")
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class transformations():

    def __init__() -> None:
        pass

    def lower(df):
        '''
        This function converts all column names to lowercase 
        Inputs: dataframe
        Return: dataframe with column names in lowercase
        '''
        df.columns = [col.lower() for col in df.columns]
        return df
    
    def target_category(x):
        '''
        This funcion categorizes each tweet based on number of reactions
        Inputs: number of total reactions
        Return: category (0 for Low, 1 for Medium and 2 for High)
        '''       
        if x < 1300:
            return 0
        if x < 4200:
            return 1
        else:
            return 2
  

class features():

    def __init__() -> None:
        pass   

    def exclamation(row):
        '''
        This function checks for exclamation mark in a dataframe columns
        Inputs: dataframe column
        Return: 0 if an exclamation mark is found, 0 if not
        '''
        if "!" in str(row) or "¡" in str(row):
            return 1
        return 0
    
    def lang_detect(text):
        '''
        This function checks for the language of the text
        Inputs: text
        Return: language of the text
        '''     
        DetectorFactory.seed = 0
        try:
            return detect(str(text))
        except:
            return 'other'
    

class text_cleaning():

    def __init__() -> None:
        pass   

    def hashtags(text): #function that removes hashtags if found in text
        regex = re.compile("(@[A-Za-z0-9-ñ]+)|(#[A-Za-z0-9-ñ]+)")
        return regex.sub('', str(text).lower())
    
    def webs(text): #function that removes web mentions if found in text
        regex = re.compile("(\w+:\/\/\S+)")
        return regex.sub('', str(text).lower())
    
    def punctuation_marks(text): #function that removes punctuation marks if found in text
        regex = re.compile("(\.)|(\;)|(\:)|(\¡)|(\¿)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
        return regex.sub('', str(text).lower())
    
    def html(text): #function that removes html if found in text
        return BeautifulSoup(text).get_text()
    
    def remove_other(text): #function that replaces common expressions if found in text
        return text.replace("\n"," ").replace('100%','muy')
    
    def remove_stopwords(df): #function that removes stopwords if found in text
        spanish_stopwords = stopwords.words('spanish')
        return " ".join([word for word in df.split() if word not in spanish_stopwords])
    

class lexicon():

    def __init__() -> None:
        pass   

    def lemma_doc(doc):
        '''
        This function lemmatizes the text
        Inputs: text
        Return: lemmatized text    
        '''
        parsed_doc = nlp(doc)
        new_doc = " ".join([token.lemma_ for token in parsed_doc])
        return new_doc       

    def score_lexicon_lemmatized(text, path, threshold = 0.2):
        '''
        This function defines the polarity of a text using a spanish lexicon 
        Inputs: 
        - text: lemmatized text - type string
        - path: path to the lexicon - type string
        - threshold: threshold that determines the minimum score to define a text as positive. Default is set to 0.2
        Return: score as "P" for positive, "N" for negative and "N/S" for the undetermined ones.    
        '''
        df_lexicon = pd.read_csv(path)

        score = 0
        cuenta = 0
        for word in text.split():
            try:
                score += df_lexicon[df_lexicon.lemma == word].Polarity.values[0]
                cuenta += 1
            except:
                continue
        score = score/cuenta if cuenta > 0 else -99999  

        return "P" if score > threshold else "N" if score > -99999 else "N/S"
    

class model():

    def __init__() -> None:
        pass   

    def gridsearch_score(prepro_pipeline, classifier, parameters, model_scoring, X_train, y_train):
        '''
        This function creates and trains a classifying model. It uses a transformation pipeline to be introduced as input, 
        it performs a cv gridsearch using the parameters introduced as input maximizing the score also introduced as input.
        Inputs: 
        - prepo_pipeline: transformation pipeline - type pipeline
        - classifier: class of the model to be used. Options are 'LGBMClassifier()', 'RandomForestClassifier()', 'XGBClassifier()' or 'KNeighborsClassifier()' - type class
        - parameters: dictionary of parameters to be used in gridsearchCV. Parameters should be passed as "model__" + parameter name - type dictionary
        - model_scoring: metric to be used in gridsearchCV, e.g. "accuracy" - type string
        - X_train: dataframe with variables to be used for training model - type dataframe, series or arrays
        - y_train: target values to be used for training model - type series or array
        Return: array with two elements:
                - trained model pipeline - type pipeline
                - message indicating the best score obtained by the trained model - type string
        '''        
        
        pipeline = Pipeline([
            ("preprocesado",prepro_pipeline),
            ('model', classifier)
        ])

        grid_search = GridSearchCV(pipeline,
                                parameters,
                                cv=5,
                                n_jobs=-1 ,
                                scoring= model_scoring)

        grid_search.fit(X_train, y_train)
        grid_model = grid_search.best_estimator_

        result =  [grid_model, str("Best "+ model_scoring+ ' for model is '+ str(round(grid_search.best_score_,2)))]
        return result   


