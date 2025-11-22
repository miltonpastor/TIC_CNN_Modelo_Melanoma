pip install --upgrade pip
pip install -r requirements.txt
python3 -m pip install tensorflow[and-cuda]
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
