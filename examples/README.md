# Examples

## Single File Inference

```bash
python scripts/infer_ogaal_ctc.py \
  --audio-path /path/to/audio.wav \
  --model-dir /path/to/model_repo
```

## Folder Inference

```bash
python scripts/infer_ogaal_ctc.py \
  --audio-dir /path/to/audio_folder \
  --recursive \
  --model-dir /path/to/model_repo
```

## Browser Demo

```bash
python scripts/web_demo.py \
  --host 127.0.0.1 \
  --port 7861 \
  --model-dir /path/to/model_repo
```
