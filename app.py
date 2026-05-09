"""YouTube Viral Prediction — Flask API (Random Forest only)"""
from flask import Flask, request, jsonify, send_from_directory
import numpy as np, pickle, json, os, time, subprocess, sys, sklearn

app = Flask(__name__, static_folder='static')
BASE       = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, 'models')

def _needs_retrain():
    ver_file = os.path.join(MODELS_DIR, 'sklearn_version.txt')
    pkl_file = os.path.join(MODELS_DIR, 'rf_model.pkl')
    if not os.path.exists(pkl_file) or not os.path.exists(ver_file):
        return True
    return open(ver_file).read().strip() != sklearn.__version__

if _needs_retrain():
    print("Retraining …")
    subprocess.run([sys.executable, os.path.join(BASE, 'train_model.py')])

with open(os.path.join(MODELS_DIR, 'rf_model.pkl'),  'rb') as f: MODEL = pickle.load(f)
with open(os.path.join(MODELS_DIR, 'metadata.json'))      as f: META  = json.load(f)
print(f"Model ready  Acc={META['accuracy']}%  AUC={META['auc']}%")

@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return r

# ── HOMEPAGE ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

# ── OPTIONS preflight (must NOT match /) ──────────────────────────────────
@app.route('/<path:path>', methods=['OPTIONS'])
def options(path):
    return ('', 204)

# ── API ───────────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'accuracy': META['accuracy'], 'auc': META['auc']})

@app.route('/api/meta')
def meta():
    return jsonify({k: META[k] for k in
        ['accuracy','f1','auc','cv_mean','cv_std','n_samples',
         'confusion_matrix','feature_importances','feature_names',
         'categories','upload_times']})

@app.route('/api/predict', methods=['POST'])
def predict():
    d = request.get_json(force=True)
    try:
        X = np.array([[
            int(d.get('category', 0)),
            float(d.get('duration', 600)),
            int(d.get('upload_time', 2)),
            float(d.get('hook', 0.5)),
            float(d.get('retention', 0.5)),
            float(d.get('engagement', 5.0)),
            int(d.get('trending', 0)),
        ]])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    t0   = time.time()
    pred = int(MODEL.predict(X)[0])
    prob = float(MODEL.predict_proba(X)[0][1])
    ms   = round((time.time() - t0) * 1000, 2)

    fi   = MODEL.feature_importances_
    top  = sorted(zip(META['feature_names'], fi), key=lambda x: -x[1])

    return jsonify({
        'viral':        bool(pred),
        'probability':  round(prob * 100, 2),
        'confidence':   round(max(prob, 1-prob) * 100, 2),
        'latency_ms':   ms,
        'top_features': [{'name': n, 'importance': round(v*100, 2)} for n,v in top],
        'tips':         _tips(prob, d),
    })

def _tips(prob, d):
    tips = []
    if float(d.get('hook', 0)) < 0.5:
        tips.append("Strengthen your hook — the first 5 seconds are critical on YouTube.")
    if float(d.get('retention', 0)) < 0.5:
        tips.append("Improve audience retention with tighter editing and pattern interrupts.")
    if float(d.get('engagement', 0)) < 5:
        tips.append("Drive engagement with CTAs, polls, and a pinned comment.")
    if not int(d.get('trending', 0)):
        tips.append("Tie content to a trending topic or hashtag to boost discoverability.")
    if float(d.get('duration', 600)) < 420:
        tips.append("Videos 7–20 min tend to maximise Watch Time and algorithm favour.")
    if int(d.get('upload_time', 2)) == 0:
        tips.append("Try uploading in the Evening — historically higher YouTube engagement.")
    if prob > 0.7:
        tips.append("Strong viral signals! Promote heavily in the first hour.")
    return tips[:4]

if __name__ == '__main__':
    print("Starting server → http://localhost:5050")
    app.run(debug=True, port=5050, host='0.0.0.0')
