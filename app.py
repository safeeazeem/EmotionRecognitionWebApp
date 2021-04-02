from flask import Flask, render_template, request
import numpy as np 
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle



app = Flask(__name__)

# loading LSTM model 
lstm_model = load_model('LSTM_Model\LSTM_Model_Emotions.h5')

# CNN Model
cnn_model = load_model('CNN_Model\CNN_Model_Emotions.h5')

with open('CNN_Model\CNN_tokenizer.pickle', 'rb') as tokenCNN:
    loaded_tokenzier_CNN = pickle.load(tokenCNN)


with open('LSTM_Model\LSTM_tokenizer.pickle', 'rb') as tokenLSTM:
    loaded_tokenzier_LSTM = pickle.load(tokenLSTM)



@app.route('/')
def home():
    # return 'Hello, World!'
    return (render_template('index.html'))


@app.route('/predict', methods = ['POST'])
def predict():
     
    #  FOR LSTM BUTTON
    if request.form.get('LSTM'):
        text = [request.form['text_input']]
        text_1 = loaded_tokenzier_LSTM.texts_to_sequences((text))
        text_2 = pad_sequences(text_1, maxlen=50, padding = 'post')
        pred = lstm_model.predict(text_2)
            
        result = np.argmax(pred)
    
        if result == 0:
            result_label ="Anger"
        elif result == 1:
            result_label="Fear"
        elif result == 2:
            result_label="Happiness"
        elif result == 3:
            result_label="Sadness"

        return(render_template('index.html', result_label=f'LSTM_Result: The Emotion is {result_label}',
                    text = f'Your Input: {text}'))
    # FOR CNN BUTTON
    elif request.form.get('CNN'):         
        text = [request.form['text_input']]
        text1 = loaded_tokenzier_CNN.texts_to_sequences((text))
        text2 = pad_sequences(text1, maxlen=50, padding = 'post')
        pred_cnn = cnn_model.predict(text2)
        results_cnn = np.argmax(pred_cnn)
        if results_cnn == 0:
            result_label_cnn ="Anger"
        elif results_cnn == 1:
            result_label_cnn="Fear"
        elif results_cnn == 2:
            result_label_cnn="Happiness"
        elif results_cnn == 3:
            result_label_cnn="Sadness"       



        return(render_template('index.html', result_label=f'CNN_RESULT: The Emotion is {result_label_cnn}',text = f'Your Input: {text}'))



  




if __name__ == "__main__":

    app.run(debug=True)
    
   