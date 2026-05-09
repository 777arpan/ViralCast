"""
YouTube Viral Prediction — Random Forest Training
Features: content_category, video_duration, upload_time,
          hook_strength, audience_retention, engagement_rate, trending_topic
"""
import numpy as np
import pandas as pd
import pickle, json, os, sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

np.random.seed(42)
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

CATEGORIES = ['Gaming', 'Education', 'Entertainment', 'Tech', 'Vlog',
              'Music', 'Sports', 'Comedy', 'News', 'Beauty']
UPLOAD_TIMES = ['Morning', 'Afternoon', 'Evening', 'Night']

# Upload-time multipliers (Evening/Night tend to perform better on YT)
UPLOAD_MULT = {'Morning': 0.85, 'Afternoon': 1.0, 'Evening': 1.15, 'Night': 1.05}
# Category multipliers
CAT_MULT = {'Gaming':1.2,'Entertainment':1.15,'Comedy':1.1,'Music':1.1,
            'Sports':1.05,'Tech':1.0,'Beauty':0.95,'Vlog':0.9,
            'Education':0.85,'News':0.8}

def generate(n=12000):
    print(f"Generating {n:,} samples …")
    category      = np.random.choice(CATEGORIES, n)
    upload_time   = np.random.choice(UPLOAD_TIMES, n)
    duration      = np.random.randint(30, 3601, n)          # 30s – 60min
    hook          = np.random.uniform(0, 1, n)               # 0–1
    retention     = np.random.uniform(0.1, 1.0, n)          # 0–100 %
    engagement    = np.random.uniform(0.1, 25.0, n)         # 0–25 %
    trending      = np.random.choice([0, 1], n, p=[0.55, 0.45])

    # Duration sweet-spot: 7–20 min scores highest
    dur_score = np.where((duration >= 420) & (duration <= 1200),
                         1.0, np.where(duration < 120, 0.6, 0.8))

    cat_w  = np.array([CAT_MULT[c]  for c in category])
    time_w = np.array([UPLOAD_MULT[t] for t in upload_time])

    score = (
        0.28 * retention
      + 0.22 * (engagement / 25)
      + 0.18 * hook
      + 0.12 * trending
      + 0.10 * dur_score
      + 0.05 * (cat_w - 0.8) / 0.4
      + 0.05 * (time_w - 0.85) / 0.3
    )
    score += np.random.normal(0, 0.04, n)
    score = np.clip(score, 0, 1)
    viral = (score > 0.55).astype(int)

    cat_enc  = np.array([CATEGORIES.index(c)    for c in category])
    time_enc = np.array([UPLOAD_TIMES.index(t)  for t in upload_time])

    df = pd.DataFrame({
        'category':    cat_enc,
        'duration':    duration,
        'upload_time': time_enc,
        'hook':        hook,
        'retention':   retention,
        'engagement':  engagement,
        'trending':    trending,
        'viral':       viral,
    })
    print(f"  Viral ratio: {viral.mean():.1%}  ({viral.sum():,} / {n:,})")
    return df

def train():
    df = generate(12000)
    df.to_csv(os.path.join(MODELS_DIR, 'dataset.csv'), index=False)

    X = df.drop('viral', axis=1).values
    y = df['viral'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, n_jobs=-1, random_state=42
    )
    print("Training Random Forest (300 trees) …")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv     = cross_val_score(model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)

    meta = {
        'accuracy':   round(accuracy_score(y_test, y_pred)         * 100, 2),
        'precision':  round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall':     round(recall_score(y_test, y_pred, zero_division=0)    * 100, 2),
        'f1':         round(f1_score(y_test, y_pred, zero_division=0)        * 100, 2),
        'auc':        round(roc_auc_score(y_test, y_prob)           * 100, 2),
        'cv_mean':    round(cv.mean() * 100, 2),
        'cv_std':     round(cv.std()  * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'feature_importances': model.feature_importances_.tolist(),
        'feature_names': ['category','duration','upload_time','hook',
                          'retention','engagement','trending'],
        'categories':   CATEGORIES,
        'upload_times': UPLOAD_TIMES,
        'n_samples':    12000,
        'sklearn_version': sklearn.__version__,
    }

    print(f"  Accuracy={meta['accuracy']}%  F1={meta['f1']}%  AUC={meta['auc']}%  CV={meta['cv_mean']}±{meta['cv_std']}%")

    with open(os.path.join(MODELS_DIR, 'rf_model.pkl'),  'wb') as f: pickle.dump(model, f)
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f: json.dump(meta, f, indent=2)
    with open(os.path.join(MODELS_DIR, 'sklearn_version.txt'), 'w') as f: f.write(sklearn.__version__)
    print("✓ Saved rf_model.pkl + metadata.json")

if __name__ == '__main__':
    train()
