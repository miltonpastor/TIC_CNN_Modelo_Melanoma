pip install --upgrade pip
pip install -r requirements.txt
pip install tensorflow[and-cuda]
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

python ./scripts/continue_training.py