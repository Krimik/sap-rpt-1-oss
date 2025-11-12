# SAP RPT Playground

Interactive playground for the `SAP/sap-rpt-1-oss` tabular in-context learner. The app combines a FastAPI backend with a React/Vite frontend so you can upload your own datasets, tune inference parameters, monitor progress in real time, and download scored results.

## Requirements

- Python 3.11
- Node.js 20+
- Hugging Face account with access to `SAP/sap-rpt-1-oss`
- `HUGGINGFACE_API_KEY` token saved locally (copy `env.example` to `.env` and fill in the value)

> Tip: accept the model’s license on Hugging Face before running the playground or checkpoint downloads will fail.

## Quickstart

1. **Set up environment**
   ```bash
   cd /path/to/SAP-RPT-1
   cp env.example .env    # edit and place your Hugging Face token
   chmod +x scripts/dev.sh
   ```

2. **Launch everything with one command**
   ```bash
   ./scripts/dev.sh
   ```
   The script:
   - loads `.env` and prepares a Python virtualenv under `playground/backend/.venv`
   - installs backend requirements (including the editable `sap-rpt-1-oss` package)
   - waits for the FastAPI health check at `http://127.0.0.1:8000/api/health`
   - installs frontend dependencies (if needed) and starts Vite on port 5173

3. **Open the UI** at [http://localhost:5173](http://localhost:5173). The Status banner shows Hugging Face auth, checkpoint cache state, and whether you are running on GPU or CPU. Progress updates stream over WebSockets and the results pane renders metrics in a table with a CSV download for the scored test split.

4. **Stop the playground** with `Ctrl+C` in the same terminal. The script cleans up the background backend process automatically.

## Example Data

Place CSV, Parquet, or JSON files inside `example_datasets/`. They are detected automatically and listed in the Dataset panel alongside manual uploads. When a run finishes you can download results named `<dataset_name> - results.csv`.

## Advanced Notes

- Health check: `http://localhost:8000/api/health`
- Result download: the UI provides a tokenized link (`/api/results/{token}/download`) for the scored test split.
- Adjust inference settings (context size, bagging, split ratio) in the Configuration panel before running a job.
