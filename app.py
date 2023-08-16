from flask import Flask, request, render_template
import pickle
from nltk.stem import WordNetLemmatizer
import re,string
# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')

vector = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

def clean_text(text): ## function to clean the data
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower() #make lower case
    text = re.sub('\[.*?\]', ' ', text) #remove text in square brackets
    text = re.sub('https?://\S+|www\.\S+', ' ', text) #remove links
    text = re.sub('@\S+', ' ', text) #remove mentions
    text = re.sub('#\S+', ' ', text) #remove hastags
    text = re.sub('<.*?>+', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) #remove punctuations
    text = re.sub('\n', ' ', text) #remove newlines
    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub('\s+', ' ', text) #remove extra white spaces

    return text

#stemming function defined
lemmatizer=WordNetLemmatizer() #initialize lemmatizing object

def lemmatize_text(text):
    '''Reduce words to their base form in a text'''
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split(' '))
    return text

# stop_words = stopwords.words('english')
# def remove_stopwords(text):
#     ''' Remove stop words from text'''
#     text = ' '.join(word for word in text.split(' ') if word not in stop_words)
#     return text


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("prediction.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        news = clean_text(news)
       # news = remove_stopwords(news)
        news = lemmatize_text(news)

        prediction_prob = model.predict_proba(vector.transform([news]))
        prediction_confidence = round(prediction_prob.max() * 100, 2)
        predict = model.predict(vector.transform([news]))[0]
        prediction_label = "True News" if predict == 0 else "Fake News"

        prediction_text = f"News headline is -> {prediction_label}, I am {prediction_confidence}% sure."

        return render_template("prediction.html", prediction_text=prediction_text)

    else:
        return render_template("prediction.html")


if __name__ == '__main__':
    app.debug = True
    print("Starting server...")
    app.run(host='0.0.0.0', port=5000)