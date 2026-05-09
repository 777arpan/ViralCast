# ViralCast — YouTube Viral Prediction
### College ML Project | Random Forest · Flask · Scikit-learn

## Quick Start

### 1. Install dependencies
```bash
pip install flask scikit-learn pandas numpy
```

### 2. Run (auto-trains on first launch ~30 sec)
```bash
python3 app.py
```
Open → http://localhost:5050

## Inputs
| Field | Type | Range |
|---|---|---|
| Content Category | Dropdown | Gaming, Education, Entertainment, Tech, Vlog, Music, Sports, Comedy, News, Beauty |
| Video Duration | Slider | 30s – 60 min |
| Upload Time | Dropdown | Morning / Afternoon / Evening / Night |
| Hook Strength | Slider | 0 – 100% |
| Audience Retention Rate | Slider | 0 – 100% |
| Engagement Rate | Slider | 0 – 25% |
| Trending Topic | Toggle | Yes / No |

## Model
- **Algorithm:** Random Forest (300 trees)
- **Dataset:** 12,000 synthetic YouTube samples
- **Accuracy:** 87.1% · **F1:** 86.7% · **AUC:** 95.6%
- **Cross-val:** 88.1% ± 0.5%

## API
| Method | Endpoint | Description |
|---|---|---|
| GET | /api/health | Status + accuracy |
| POST | /api/predict | Prediction + tips |
| GET | /api/meta | Full model metadata |
