# Model Weights

Download every registered model:

```bash
python models/download_all.py
```

Download one model:

```bash
python models/download_weights.py --model xception
```

DeepGuard v2.0 supports:

- `efficientnet_b4`
- `xception`
- `vit_base`
- `resnet50_ffpp`
- `efficientnet_b7`
- `mesonet4`

Notes:

- The downloader first tries the deepfake-specific sources from the project brief.
- `resnet50_ffpp` uses a public FF++ C23 checkpoint from the PyDeepFakeDet model zoo on Google Drive.
- If a published checkpoint is unavailable, the script falls back to a generic pretrained model when the brief allows it and marks that payload as a fallback.
- `mesonet4` attempts a best-effort Keras-to-PyTorch conversion. If that fails, DeepGuard saves a random-weight fallback and marks the model as needing fine-tuning.
