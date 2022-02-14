--DL for HAR

Author: Kartikay Srivastava
Email: kartikay.srivastava@student.uni-siegen.de

Structure of directory:

1. main.py file

This file contains the follwoing funtions:

a. load_data: to load data for intepretation.
b. resampling: to downsample and upsample datasets
c. evaulation: to create plots for predition results.
d. plot_with_sensor: to create plots of sensor data with bit maps pf ground truth and predicted values

2. data folder: currently empty as Git is not allowing to upload large files.

3. preditions folder:

a. data_15_person: contains data for the 15th sample at various sampling rates (50 Hz,25 Hz, 12.5 Hz etc).
b. plots_sensor: contains plots of sensor data along with bit maps of ground truth and predicted values.
c. prediction_results: contains csv files for predicted vs ground truth values as predicted by dl-har algorithm.
d. f1_final.csv: contains f1 scores for different classes (eg running, sitting, standing) at various sampling rates (50 Hz, 25 Hz,12.5 Hz etc)

## Comments will be later added to main.py file. 