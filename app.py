
from multiprocessing.resource_tracker import getfd
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import logging
import joblib
import json
import sys
import os
import sqlite3
import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, g
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

current_dir = os.path.dirname(__file__)

# Flask app
app = Flask(__name__, static_folder='', template_folder='template')
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key_here'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Logging
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

import sqlite3

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL,
                        mobile TEXT,
                        dob TEXT,
                        location TEXT
                    )''')

    # Predictions table
    cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        name TEXT,
                        gender TEXT,
                        education TEXT,
                        self_employed TEXT,
                        marital_status TEXT,
                        dependents TEXT,
                        applicant_income REAL,
                        coapplicant_income REAL,
                        loan_amount REAL,
                        loan_term REAL,
                        credit_history REAL,
                        property_area TEXT,
                        result TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )''')

    # Documents table
    cursor.execute('''CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        aadhaar TEXT NOT NULL,
                        pan TEXT NOT NULL,
                        voter_id TEXT NOT NULL,
                        ration_card TEXT NOT NULL
                      )''')

    conn.commit()
    conn.close()

init_db()

DATABASE = "users.db"

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row  # Enables column name-based access
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# Function
def ValuePredictor(data=pd.DataFrame):
    # Model name
    model_name = r'xgboostModel.pkl'
    # Directory where the model is stored
    model_dir = os.path.join(current_dir, model_name)
    # Load the model
    loaded_model = joblib.load(open(model_dir, 'rb'))
    # Predict the data
    result = loaded_model.predict(data)
    return result[0]


app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# Home page
@app.route('/index')
def home():
    return render_template('index.html')

# Login page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')

    return render_template('login.html')


ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the credentials match
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Login successful! Welcome to the Admin Dashboard.', 'success')
            return redirect(url_for('view_documents'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('admin_login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        mobile = request.form['mobile']
        dob = request.form['dob']
        location = request.form['location']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password, mobile, dob, location) VALUES (?, ?, ?, ?, ?)",
                           (username, password, mobile, dob, location))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists. Please choose another.', 'danger')
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please log in to view your dashboard.', 'warning')
        return redirect(url_for('login'))

    # Get the logged-in user's username
    username = session['username']
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Fetch all users' data
    cursor.execute("SELECT id, username, mobile, dob, location FROM users")
    all_users_data = cursor.fetchall()

    if not all_users_data:
        flash('No users found!', 'danger')
        return redirect(url_for('login'))

    # Prepare to collect predictions for all users
    users_with_predictions = []

    for user in all_users_data:
        user_id, username, mobile, dob, location = user

        # Fetch user's predictions
        cursor.execute("SELECT name, gender, education, self_employed, marital_status, dependents, "
                       "applicant_income, coapplicant_income, loan_amount, loan_term, credit_history, "
                       "property_area, result, timestamp FROM predictions WHERE user_id = ?", (user_id,))
        predictions_data = cursor.fetchall()

        # Add user data and their predictions to the list
        users_with_predictions.append({
            'user': user,
            'predictions': predictions_data
        })

    conn.close()

    return render_template('dashboard.html', users_with_predictions=users_with_predictions)



@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/prediction', methods=['POST'])
def predict():
    if 'username' not in session:
        flash('Please log in to access the prediction feature.', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = session['username']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Fetch user ID
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        user_id = cursor.fetchone()[0]

        # Get data from the form
        name = request.form['name']
        gender = request.form['gender']
        education = request.form['education']
        self_employed = request.form['self_employed']
        marital_status = request.form['marital_status']
        dependents = request.form['dependents']
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = float(request.form['loan_term'])
        credit_history = float(request.form['credit_history'])
        property_area = request.form['property_area']

        # Prediction process
        schema_name = 'data/columns_set.json'
        schema_dir = os.path.join(current_dir, schema_name)
        with open(schema_dir, 'r') as f:
            cols = json.loads(f.read())
        schema_cols = cols['data_columns']

        # Update schema columns with input values
        schema_cols.update({
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Gender_Male': gender,
            'Married_Yes': marital_status,
            'Education_Not Graduate': education,
            'Self_Employed_Yes': self_employed,
            'Credit_History_1.0': credit_history,
        })

        df = pd.DataFrame(data={k: [v] for k, v in schema_cols.items()}, dtype=float)
        result = ValuePredictor(data=df)

        prediction = 'Approved' if int(result) == 1 else 'Rejected'

        # Store prediction in the database
        cursor.execute('''INSERT INTO predictions 
                          (user_id, name, gender, education, self_employed, marital_status, dependents, 
                          applicant_income, coapplicant_income, loan_amount, loan_term, credit_history, 
                          property_area, result) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (user_id, name, gender, education, self_employed, marital_status, dependents,
                        applicant_income, coapplicant_income, loan_amount, loan_term, credit_history,
                        property_area, prediction))
        conn.commit()
        conn.close()

        # Return the prediction
        flash(f'Prediction: {prediction}', 'info')
        return render_template('prediction.html', prediction=prediction)

    return render_template('error.html', prediction='An error occurred.')



@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Retrieve files from the form
        aadhaar = request.files.get('aadhaar')
        pan = request.files.get('pan')
        voter_id = request.files.get('voter_id')
        ration_card = request.files.get('ration_card')

        # Check if all files are provided
        if not all([aadhaar, pan, voter_id, ration_card]):
            flash("All documents are required!", "error")
            return redirect(url_for('upload_file'))

        # Secure filenames
        aadhaar_filename = secure_filename(aadhaar.filename)
        pan_filename = secure_filename(pan.filename)
        voter_id_filename = secure_filename(voter_id.filename)
        ration_card_filename = secure_filename(ration_card.filename)

        # Ensure upload directory exists
        upload_path = app.config['UPLOAD_FOLDER']
        os.makedirs(upload_path, exist_ok=True)

        # Save files
        try:
            aadhaar.save(os.path.join(upload_path, aadhaar_filename))
            pan.save(os.path.join(upload_path, pan_filename))
            voter_id.save(os.path.join(upload_path, voter_id_filename))
            ration_card.save(os.path.join(upload_path, ration_card_filename))
        except Exception as e:
            flash(f"Error saving files: {str(e)}", "error")
            return redirect(url_for('upload_file'))

        # Save file names in the database
        try:
            conn = get_db()
            with conn:
                conn.execute('''INSERT INTO documents (aadhaar, pan, voter_id, ration_card)
                                VALUES (?, ?, ?, ?)''',
                             (aadhaar_filename, pan_filename, voter_id_filename, ration_card_filename))
            flash("Files uploaded successfully!", "success")
            return redirect(url_for('predict'))
        except Exception as e:
            flash(f"Database error: {str(e)}", "error")
            return redirect(url_for('upload_file'))

    return render_template('upload.html')



@app.route('/documents')
def view_documents():
    """Show Uploaded Documents"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''SELECT id, aadhaar, pan, voter_id, ration_card FROM documents''')
    
    documents = cursor.fetchall()
    conn.close()

    return render_template('documents.html', documents=documents)



if __name__ == '__main__':
    app.run(debug=True)
