from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
#load the model
model = pickle.load(open('saved_model.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())


@app.route('/predict', methods=['POST', 'GET']) 
def predict():
    if request.method == 'POST':
        try:
            sepal_length = float(request.form['SepalLengthCm'])
            sepal_width = float(request.form['SepalWidthCm'])
            petal_length = float(request.form['PetalLengthCm'])
            petal_width = float(request.form['PetalWidthCm'])
        except KeyError:
            # Handle missing form fields
            return render_template('index.html', result="Please provide all input fields.")
        
        predicted_class = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        return render_template('index.html', result=predicted_class)
    else:
        # Handle GET requests
        return render_template('index.html', result="Submit a form to make a prediction.")



if __name__ == '__main__':
    app.run(debug=True)   
    