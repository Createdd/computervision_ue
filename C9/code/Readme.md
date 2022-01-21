


# Code usage of Submission
- Copy original images as `data` folder into `C9`
- Navigate into the folder `C9/code` and execute scripts from there
- Execute `python detect.py --set validation --dataset_root ../data` to get detections for a dataset and save them as json files, run detect.py from the console and specify the dataset name and data path, e.g.:
- Execute `python evaluate.py ../val.json --set validation --dataset_root ../data` to evaluate a set of detections (specify the path to the json file, the dataset name and data path)
- Execute `visualization.ipynb` to visualize any intermediate steps of the image processing and visually compare detections and targets
- Execute `anomaly_detection_autoencoder_SAR_JKU.ipynb` to navigate to link to Google Colab notebook
