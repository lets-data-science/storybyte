# StoryByte: reproduce the shipped model end to end.
PY=python3
all: data train export evaluate
data: download-reference tokenizer prepare
download-reference: ; $(PY) scripts/01_download_data.py --subset_mb 400
download-full: ; $(PY) scripts/01_download_data.py
tokenizer: ; $(PY) scripts/02_train_tokenizer.py
prepare: ; $(PY) scripts/03_prepare_data.py
train: ; $(PY) scripts/04_train.py
export: ; $(PY) scripts/05_export_artifacts.py
evaluate: ; $(PY) scripts/06_evaluate.py
generate: ; $(PY) scripts/reference_forward.py "Once upon a time"
verify: ; $(PY) scripts/07_verify_repository.py
clean: ; rm -f data/*.bin checkpoints/*.pt
.PHONY: all data download-reference download-full tokenizer prepare train export evaluate generate verify clean
