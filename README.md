# DeepLearningProject – UrbanSound8K
## Setup (Ubuntu/WSL)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

## Run
python -m src.check_data
python -m src.make_subset
python -m src.train
python -m src.evaluate
python -m src.infer
