--DL for HAR

Author: Kartikay Srivastava
Email: kartikay.srivastava@student.uni-siegen.de

Structure of directory:

1. main.py file

This file contains the following functions:

a. load_data: to load data for intepretation.
b. resampling: to downsample and upsample datasets
c. evaluation: to create plots for prediction results.
d. plot_with_sensor: to create plots of sensor data with bit maps pf ground truth and predicted values

2. data folder: contains originial rwhar dataset used for interpretation and analysis.

3. predictions folder:

a. data_15_person: contains data for the 15th sample at various sampling rates (50 Hz,25 Hz, 12.5 Hz etc).
b. plots_sensor: contains plots of sensor data along with bit maps of ground truth and predicted values.
c. prediction_results: contains csv files for predicted vs ground truth values as predicted by dl-har program.
d. f1_final.csv: contains f1 scores for different classes (eg running, sitting, standing) at various sampling rates (50 Hz, 25 Hz,12.5 Hz etc)

# comments have also been added to main file
