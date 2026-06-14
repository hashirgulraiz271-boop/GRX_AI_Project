GRX AI: Smart Route & Rider Safety Assistant
==============================================
AI Lab Term Project Spring 2026
Members: Hashir Gulraiz (L1F23BSCS1100) | Huzaifa Qadeer (L1F23BSCS0069)

HOW TO RUN:
-----------
1. Install dependencies:
   pip install -r requirements.txt

2. Train the model (first time only):
   cd model
   python train_model.py

3. Run the Flask app:
   python app.py

4. Open browser at:
   http://localhost:5000

PROJECT STRUCTURE:
------------------
app.py                  - Flask backend
templates/index.html    - Frontend UI
model/train_model.py    - Model training script
model/knn_model.pkl     - Trained KNN model
model/scaler.pkl        - StandardScaler
model/encoders.pkl      - Feature encoders
model/grx_dataset.csv   - Dataset (3000 samples)
static/images/          - Performance charts
database/grx_ai.db      - SQLite database
GRX_AI_Model.ipynb      - Jupyter notebook
