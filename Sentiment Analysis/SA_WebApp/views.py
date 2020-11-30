
from pyramid.view import (
    view_config,
    view_defaults
    )
from pyramid.response import Response

from SA_WebApp.SA_models.text_process import text_preprocessing, createAvgWordVector # own class
import gensim
import keras
import numpy as np


# First view, available at http://localhost:6543/

@view_defaults(renderer='home.pt')
class TutorialViews:
    def __init__(self, request):
        self.request = request
        self.new_w2v_model = gensim.models.Word2Vec.load(r'SA_WebApp/SA_models/w2vmodel')
        self.reconstructed_model = keras.models.load_model(r"SA_WebApp/SA_models/model1")
        #print("\nloading successful\n")


    @view_config(route_name='home')
    def home(self):
        request = self.request
        new_w2v_model = self.new_w2v_model
        reconstructed_model = self.reconstructed_model
        login_url = request.route_url('home')
        referrer = request.url
        
        if referrer == login_url:
            referrer = '/'  # never use logi
        came_from = request.params.get('came_from', referrer)

        final_text = ""  
        rawtexts = ""
        processed_text = ""

        if 'form.submitted' in request.params:
            rawtexts = request.params['rawtext']
            if rawtexts != "":

                # preprocessing the Input Text
                processed_text = text_preprocessing(rawtexts)
                
                if processed_text != "":
                    split_text = processed_text.split()

                    # converting words to vector
                    text_Avg_w2v = createAvgWordVector(split_text, new_w2v_model)
                    text_Avg_w2v = np.array(text_Avg_w2v).reshape(1, 50) # converting it to 2D

                    # to load model https://www.tensorflow.org/guide/keras/save_and_serialize
                    #reconstructed_model = keras.models.load_model(r"SA_WebApp/SA_models/model1")
                    result = reconstructed_model.predict_classes(text_Avg_w2v)
                    
                    if result == 1:
                        final_text = "Positive"
                    else:
                        final_text = "Negative"
               
                else:
                    final_text = "Netural"
            else:  
                final_text = "Empty String, please enter some text" 

        return {'output': final_text , 'text': rawtexts , 'processed': processed_text}

   