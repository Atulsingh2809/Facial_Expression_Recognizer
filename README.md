# Facial Expression Recognition (FER)

![Placeholder: add your UI screenshot here](https://via.placeholder.com/800x450/1a2332/22d3ee?text=FER+Web+App+Screenshot)

End-to-end web app that classifies **seven** facial expressions (**angry**, **disgust**, **fear**, **happy**, **neutral**, **sad**, **surprise**) from a **webcam** stream or **uploaded** image. A **custom CNN** is trained on **FER2013** (48×48 grayscale); the **Flask** API serves predictions and the **React** client provides a dark, responsive UI.

**Target:** ≥ **80%** test accuracy on FER2013 (Private Test) with the provided architecture, augmentation, and training schedule — run `evaluate.py` after training to record your exact numbers.

---

## Architecture (ASCII)

```
┌─────────────────┐     HTTPS / JSON      ┌──────────────────────────┐
│  React +        │  POST /predict         │  Flask (Gunicorn)       │
│  Tailwind       │  (multipart image) ────►│  OpenCV Haar → 48×48    │
│  Webcam / Upload│◄───────────────────────│  TensorFlow/Keras CNN    │
└─────────────────┘     emotion + probs     └───────────┬──────────────┘
                                                        │
                                                        ▼
                                            ┌──────────────────────────┐
                                            │  fer_model.h5 +          │
                                            │  label_map.json          │
                                            └──────────────────────────┘
```

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-headless-5C3EE8?logo=opencv&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3-06B6D4?logo=tailwindcss&logoColor=white)
![Render](https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render&logoColor=black)

---

## FER2013 dataset (Kaggle)

1. Create a [Kaggle](https://www.kaggle.com/) account and download an API token (`kaggle.json`).
2. Install the CLI: `pip install kaggle`
3. On Windows, put `kaggle.json` in `%USERPROFILE%\.kaggle\`; on Linux/macOS, `~/.kaggle/kaggle.json` (chmod 600).
4. Download the dataset:

```bash
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip
# You should have fer2013.csv (columns: emotion, pixels, Usage)
```

---

## Local setup

### Conda environment (recommended, Python 3.10)

```bash
conda create -n fer python=3.10 -y
conda activate fer
cd facial-expression-recognition/backend
pip install -r requirements.txt
```

### Train the model

Place `fer2013.csv` in `backend/model/` (or pass `--csv`).

```bash
cd facial-expression-recognition/backend/model
python train.py --csv /path/to/fer2013.csv --out-dir .
```

This writes **`fer_model.h5`**, **`training_metrics.json`**, and checkpoints the best weights during training.

### Evaluate (metrics + confusion matrix)

```bash
cd facial-expression-recognition/backend/model
python evaluate.py --csv /path/to/fer2013.csv --model fer_model.h5 --label-map label_map.json
```

The script prints training / validation / test accuracy, a per-class report, saves **`confusion_matrix.png`**, and ends with:

`Test Accuracy: XX.XX%`

### Run the backend

From `facial-expression-recognition/backend`:

```bash
set FLASK_APP=app.py
# Linux/macOS: export FLASK_APP=app.py
flask run --host 0.0.0.0 --port 5000
```

Ensure `model/fer_model.h5` exists (or set `MODEL_PATH`). Production-style:

```bash
gunicorn app:app --workers 2 --bind 0.0.0.0:5000
```

### Run the frontend

```bash
cd facial-expression-recognition/frontend
set REACT_APP_API_URL=http://127.0.0.1:5000
npm install
npm start
```

---

## Model accuracy (fill after training)

| Split      | Accuracy (example — replace with your `evaluate.py` output) |
|-----------|-------------------------------------------------------------|
| Training  | _run evaluate.py_                                           |
| Validation| _run evaluate.py_                                           |
| Test      | **Target ≥ 80%** on FER2013 Private Test                    |

---

## API usage

### Health

```bash
curl -s https://YOUR-BACKEND.onrender.com/health
```

### Predict (multipart file field `image`)

```bash
curl -s -X POST -F "image=@photo.jpg" https://YOUR-BACKEND.onrender.com/predict
```

Example JSON (face found):

```json
{
  "emotion": "happy",
  "confidence": 0.94,
  "all_probabilities": {
    "angry": 0.01,
    "disgust": 0.00,
    "fear": 0.02,
    "happy": 0.94,
    "neutral": 0.02,
    "sad": 0.01,
    "surprise": 0.00
  },
  "face_detected": true,
  "face_box": { "x": 120, "y": 80, "width": 200, "height": 200 }
}
```

No face (HTTP 200):

```json
{
  "face_detected": false,
  "emotion": null,
  "confidence": null,
  "all_probabilities": null,
  "face_box": null
}
```

---

## Deploy on Render

### 1. Push this repo to GitHub

Include **`backend/model/fer_model.h5`** after training (large file — consider [Git LFS](https://git-lfs.com/) or uploading the file via Render shell / external storage; for a quick demo, commit the trained weights if size allows).

### 2. Backend — Web Service

1. In [Render Dashboard](https://dashboard.render.com/), **New +** → **Web Service**, connect the repo.
2. **Root Directory:** `facial-expression-recognition/backend` (adjust if your paths differ).
3. **Runtime:** Python (use `runtime.txt` → 3.10.x).
4. **Build command:** `pip install -r requirements.txt`
5. **Start command:** `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT`
6. **Health check path:** `/health`
7. **Environment variables** (Environment → Environment Variables):
   - `FLASK_ENV` = `production`
   - `MODEL_PATH` = `model/fer_model.h5` (default layout if the file is in `backend/model/`)
   - `CORS_ORIGINS` = your **frontend URL** (e.g. `https://fer-frontend.onrender.com`) or `*` for testing only
8. Deploy and copy the service URL, e.g. `https://fer-api-xxxx.onrender.com`.

You can also import **`backend/render.yaml`** as a Blueprint and then set **Root Directory** and secrets in the UI.

### 3. Frontend — Static Site

1. **New +** → **Static Site**, same repo.
2. **Root Directory:** `facial-expression-recognition/frontend`
3. **Build command:** `npm install && npm run build`
4. **Publish directory:** `build` (Render’s field is **Publish directory** / `staticPublishPath` in YAML)
5. **Environment variable:**
   - `REACT_APP_API_URL` = your **backend URL** with **no trailing slash**, e.g. `https://fer-api-xxxx.onrender.com`  
   This is baked in at **build time**; after changing it, trigger a **manual redeploy** of the static site.

### 4. Link frontend to backend

1. Deploy the **backend** first; copy its public `https://...` URL.
2. Set **`REACT_APP_API_URL`** on the **static site** to that URL.
3. Redeploy the frontend.
4. Set **`CORS_ORIGINS`** on the backend to the static site URL (comma-separated if multiple).

### 5. SPA routing

If client-side routes are added later, configure a **rewrite**: `/*` → `/index.html`. The included `frontend/render.yaml` lists a rewrite rule; if your Render plan/UI differs, add the same rule under the static site’s **Redirects/Rewrites**.

---

## Project layout

```
facial-expression-recognition/
├── backend/
│   ├── app.py
│   ├── model/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── fer_model.h5      # produced by training
│   │   └── label_map.json
│   ├── utils/
│   │   ├── preprocess.py
│   │   └── predict.py
│   ├── requirements.txt
│   ├── runtime.txt
│   └── render.yaml
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── tailwind.config.js
│   └── render.yaml
└── README.md
```

---

## License

Use and modify for your own projects; ensure compliance with FER2013 / Kaggle terms when using the dataset.
