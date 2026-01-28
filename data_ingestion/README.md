# data_ingestion/

This stage is responsible for acquiring raw inputs and materializing them into `.cache/`, which is the **single source of truth** for raw data.

Rules:
- `past_work/` (legacy) and remote URLs are **sources for ingestion only**.
- All downstream stages (`preprocessing/`, `model_training/`, `evaluation/`) must read raw inputs **only from `.cache/`**.

Run ingestion per pipeline:
- `python data_ingestion/sentiment_embeddings/run.py`
- `python data_ingestion/asr_commands/run.py`
- `python data_ingestion/clip_multimodal/run.py`

Configuration:
- Each ingestion script reads a per-pipeline TOML file:
	- `data_ingestion/sentiment_embeddings/config.toml`
	- `data_ingestion/asr_commands/config.toml`
	- `data_ingestion/clip_multimodal/config.toml`
- You can override the config path with `--config` and override cache root with `--cache-root`.
