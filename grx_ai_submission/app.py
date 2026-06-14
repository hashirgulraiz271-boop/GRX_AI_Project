# Triggering production build v1.0
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
import sqlite3
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ─── Load Model Assets With Absolute Directories ─────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, 'model/knn_model.pkl'), 'rb') as f:
    knn = pickle.load(f)
with open(os.path.join(BASE, 'model/scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(BASE, 'model/encoders.pkl'), 'rb') as f:
    encoders = pickle.load(f)

# ─── Production Ready Database Configuration ─────────────────────────────────
# Serverless platforms like Vercel have a read-only root system. 
# We target the writeable '/tmp' directory so SQLite can initialize.
if os.environ.get("VERCEL"):
    DB_PATH = '/tmp/grx_ai.db'
else:
    DB_PATH = os.path.join(BASE, 'database/grx_ai.db')
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            weather     TEXT,
            traffic     TEXT,
            time_of_day TEXT,
            accident    TEXT,
            distance    INTEGER,
            route_type  TEXT,
            prediction  TEXT,
            confidence  REAL,
            timestamp   TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Safe Database initialization invocation
try:
    init_db()
except Exception:
    pass

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json() or {}
        weather    = data.get('weather')
        traffic    = data.get('traffic')
        time_day   = data.get('time_of_day')
        accident   = data.get('accident_risk')
        distance   = int(data.get('distance', 10))
        route_type = data.get('route_type')

        # Feature Mapping Matrix Lookups
        w_enc  = encoders['weather'].get(weather, 0)
        tr_enc = encoders['traffic'].get(traffic, 0)
        t_enc  = encoders['time'].get(time_day, 0)
        ac_enc = encoders['accident'].get(accident, 0)
        rt_enc = encoders['route_type'].get(route_type, 0)

        features = np.array([[w_enc, tr_enc, t_enc, ac_enc, distance, rt_enc]])
        feat_df = pd.DataFrame(features, columns=['W_num','Tr_num','T_num','Ac_num','Distance','Rt_num'])
        features_scaled = scaler.transform(feat_df)

        pred_num = knn.predict(features_scaled)[0]
        pred_proba = knn.predict_proba(features_scaled)[0]
        confidence = float(max(pred_proba)) * 100

        prediction = encoders['label'].get(pred_num, 'Unknown')
        formatted_prediction = prediction.replace('_', ' ')

        # Safe DB insertion block
        try:
            db = get_db()
            db.execute('''
                INSERT INTO predictions (weather, traffic, time_of_day, accident, distance, route_type, prediction, confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (weather, traffic, time_day, accident, distance, route_type,
                  formatted_prediction, confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            db.commit()
        except Exception:
            pass

        tips = {
            'Safe_Route':    'Great! This route appears safe. Enjoy your ride!',
            'Moderate_Risk': 'Be cautious! Some risks detected. Stay alert and follow traffic rules.',
            'High_Risk':     'Danger! High risk route detected. Consider an alternative path or wait.'
        }

        return jsonify({
            'success': True,
            'prediction': formatted_prediction,
            'prediction_raw': prediction,
            'confidence': round(confidence, 1),
            'tip': tips.get(prediction, ''),
            'probabilities': {
                'Safe Route':    round(float(pred_proba[2])*100, 1),
                'Moderate Risk': round(float(pred_proba[1])*100, 1),
                'High Risk':     round(float(pred_proba[0])*100, 1)
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        db = get_db()
        rows = db.execute('SELECT * FROM predictions ORDER BY id DESC').fetchall()
        return jsonify([dict(r) for r in rows])
    except Exception:
        return jsonify([])

@app.route('/stats', methods=['GET'])
def stats():
    try:
        db = get_db()
        total = db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        safe = db.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Safe Route'").fetchone()[0]
        mod = db.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Moderate Risk'").fetchone()[0]
        high = db.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'High Risk'").fetchone()[0]
        return jsonify({'total': total, 'safe': safe, 'moderate': mod, 'high': high})
    except Exception:
        return jsonify({'total': 0, 'safe': 0, 'moderate': 0, 'high': 0})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)