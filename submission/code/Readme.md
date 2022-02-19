# Usage of submitted code
- To get detections for a dataset and save them as a json file, run `detect.py` and specify the dataset name (argument: `--set`, default: `validation`) and data path (argument: `--dataset_root`, default: `../data`). Note that the json file will be saved neighbouring to and outside of the working directory. To save the json file in the `C9` folder and use the default setting for the dataset root, copy the `data` folder containing the raw image data into `C9` and execute the script from the `C9/code` directory. <br>
E.g.: `python detect.py --set validation --dataset_root ../data` <br>
- To evaluate a set of detections and calculate the average precision compared to target labels, run `evaluate.py`, which was provided by the professors of the course "UE Computer Vision" and define the path to the detection json file, the dataset name (argument: `--set`, default: `validation`) and the data path (argument: `--dataset_root`, default: `./data`). <br>
E.g.: `python evaluate.py ../val.json --set validation --dataset_root ../data` <br>
- The script `utils.py` contains all functions used in `detect.py` and `evaluate.py`. <br>
- To plot and/or save any images from intermediate steps of the detection pipeline including labels for detections and/or targets, follow the instructions in the jupyter notebook `visualization.ipynb`. <br>



# Autoencder

During the course we also worked out an Autoencoder solution that was not incorporated in the final pipeline because the color channel variance solution provided good results for the problem design of the course. 

To see its implementation
- navigate to https://github.com/Createdd/computervision_ue to see the whole project overview or
- navigate to `anomaly_detection_autoencoder_SAR_JKU.ipynb` that contains the link to the Google Colab notebook for the autoencoder implementation
