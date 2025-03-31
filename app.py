from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('crime_model.pkl')

def normalize_prediction(prediction, min_value=0, max_value=50000, new_min=1, new_max=100):
    normalized = ((prediction - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
    return max(new_min, min(new_max, normalized))  # Ensures the value stays within 1-100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        city = request.form['city']
        crime_type = request.form['crime_type']
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hour'])
        
        input_data = pd.DataFrame([[city, crime_type, year, month, day, hour]],
                                columns=['City', 'Crime Description', 'Year', 'Month', 'Day', 'Hour'])
        
        raw_prediction = model.predict(input_data)[0]
        normalized_prediction = normalize_prediction(raw_prediction)
        
        return render_template('result.html', prediction=round(normalized_prediction, 2))
    
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        rating = request.form['rating']
        comments = request.form['comments']
        
        # Store feedback (you can add database/logging here)
        print(f"New Feedback: {name} ({email}) rated {rating}/5. Comments: {comments}")
        
        return render_template('feedback.html', 
                            success=True,
                            message="Thank you for your feedback!")

if __name__ == '__main__':
    app.run(debug=True)

