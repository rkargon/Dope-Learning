# "Dope Learning" – Generative Music using LSTMs
Music generation using deep learning and tensorflow
We implemented an LSTM over MIDI note events.

This project requires Tensorflow and the Python-MIDI library.

## Installation

Run `pip install -r requirements.txt` to install required packages.

## Running

Run `python music_model.py --train <TRAINING FILES>   --model_save_path <WHERE TO SAVE MODEL> -o <GENERATED MUSIC FILENAME>`
to train a model on the given data, and to generate a music file given the first note of the training data as context.

## Summary of files
 - music_model.py - File for training and running model
 - preprocess.py - File for reprocessing midi data
 - note_stats.py - File for analytics on MIDI tracks
 - schoenberg,py - File for some more metrics on generated music data. 
