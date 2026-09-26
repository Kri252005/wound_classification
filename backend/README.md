# WoundScope ML backend

Place the user-supplied, private artifacts in `backend/model/`:

```
model.weights.h5
config.json
metadata.json
```

They are deliberately not part of the frontend or this repository's public API. The service reconstructs the saved Keras graph from a complete config when available; otherwise it builds the exact supplied DenseNet169 training architecture and loads the supplied weights. It never uses a fake or randomly initialized classifier.

## Run

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Run from the project root instead if `backend.main` cannot be imported:

```powershell
py -m uvicorn backend.main:app --reload --port 8000
```

`GET /health` reports whether model startup succeeded. `POST /predict` expects a `file` multipart field with JPG/JPEG/PNG image data.

```powershell
curl.exe -F "file=@C:\path\to\test-image.jpg" http://localhost:8000/predict
```
