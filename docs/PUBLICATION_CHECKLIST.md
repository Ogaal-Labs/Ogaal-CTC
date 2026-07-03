# Publication Checklist

## Before Publishing

- confirm the final public model name is `Ogaal CTC`
- confirm the Hugging Face repo id
- confirm the GitHub repo name
- confirm the license after data review
- confirm there is no sensitive local path leakage in generated docs or manifests
- confirm the model card clearly says the published decode path is `transcript_only`

## Build

- run `build_release_artifacts.py`
- inspect `artifacts/hf_model_repo/README.md`
- inspect `artifacts/github_repo/README.md`
- smoke-test the CLI with `--model-dir`
- smoke-test the browser demo locally

## Hugging Face

- create the model repo
- upload the built `hf_model_repo/` folder
- verify `README.md` renders correctly
- verify `model.safetensors` and `.binary` files are stored with LFS
- verify the fixed-system JSON uses only relative asset paths

## GitHub

- publish `artifacts/github_repo/`
- verify the repo README points to the correct Hugging Face model repo
- verify there are no large scratch outputs in the repo
- verify the CLI and demo instructions use relative paths only

## Release Messaging

- say the system was intentionally trained for Somali speech recognition
- cite the published `transcript_only` metrics
- say the training effort totals roughly 72 hours of Somali speech
- mention the private multi-speaker prompt collection method without exposing private data
