PYTHON ?= python
MODEL_DIR ?= model
PORT ?= 7861
AUDIO ?=

.PHONY: check install demo infer

install:
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) -m py_compile scripts/*.py

demo:
	$(PYTHON) scripts/web_demo.py --host 127.0.0.1 --port $(PORT) --model-dir $(MODEL_DIR)

infer:
	@if [ -z "$(AUDIO)" ]; then echo "Set AUDIO=/path/to/audio.wav"; exit 1; fi
	$(PYTHON) scripts/infer_ogaal_ctc.py --audio-path $(AUDIO) --model-dir $(MODEL_DIR)
