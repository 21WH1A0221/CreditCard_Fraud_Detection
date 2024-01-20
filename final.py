import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getprediction', methods=['POST'])
def getprediction():
    input_values = [float(x) for x in request.form.values()]
    final_input = [np.array(input_values)]
    prediction = model.predict(final_input)

    # Mapping dictionary
    prediction_labels = {0: 'Valid', 1: 'Fraud'}

    # Get the corresponding label
    output_label = prediction_labels.get(prediction[0], 'Unknown')

    return render_template('index.html', output='Predicted Transition: {}'.format(output_label))


if __name__ == "__main__":
    app.run(debug=True)
