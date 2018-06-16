# Drake-LSTM

a Keras neural network trained to write new Drake lyrics

## Requirements

```
pip install tensorflow
pip install keras
pip install h5py
pip install Flask
pip install Flask-wtf
pip install gunicorn
```

## Training the network

```
python network/train.py
```

The weights will be checkpointed as hdf5 files with the format `weights-{epoch:02d}-{loss:.3f}.hdf5` and the model will be dumped as `model.yaml`. If you wish to use a different corpus, just drop it in & edit `network/train.py`.

## Generating text
Edit `network/generate.py` to use your new weights and model if desired, then:

```
python network/generate.py
```
