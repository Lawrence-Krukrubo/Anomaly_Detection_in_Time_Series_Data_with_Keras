# Anomaly_Detection_with_Time_Series_Data

In data mining, **anomaly detection (also outlier detection)** is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data. Typically the anomalous items will translate to some kind of problem such as bank fraud, a structural defect, medical problems or errors in a text. Anomalies are also referred to as _outliers, novelties, noise, deviations and exceptions_.

In this project, I will build an Anomaly Detection Model from scratch, using Deep Learning. Specifically, I will be designing and training an **LSTM autoencoder** using the **Keras API with Tensorflow 2 as the backend** to detect anomalies (sudden price changes) in the S&P 500 index. I will also create interactive charts and plots using Plotly, Python and Seaborn for data visualization and display the results in Jupyter notebooks.

<p align="center">
  <img src="https://github.com/Lawrence-Krukrubo/Anomaly_Detection_in_Time_Series_Data_with_Keras/blob/master/real-time-anomaly-detection.jpg?raw=true">
</p>

## Note:
**To fully visualize the interactive plots and charts in Plotly, its best to open the project Notebook in Google Colab.
Click on the `open in colab` button on in the Notebook. Colab is a free Interactive Development Environment (IDE) that powers your Notebook with free GPU and TPU from Google.**

## Key Concepts
* Build an LSTM Autoencoder in Keras
* Detect anomalies with Autoencoders in time series data
* Create interactive charts and plots with Plotly and Seaborn

### Project Structure:
This project is split into 8 parts that lead from one to another in a flow-fashion. Let's take a look at each part.

**Task 1: Project Overview and Import Libraries:**<br>
This project is about Anomaly Detection in Time Series Data with Keras and Tensorflow. To follow along kindly import the following libraries
* `import numpy as np`
* `import tensorflow as tf`
* `import pandas as pd`
* `pd.options.mode.chained_assignment = None`
* `import seaborn as sns`
* `from matplotlib.pylab import rcParams`
* `import matplotlib.pyplot as plt`
* `import plotly.express as px`
* `import plotly.graph_objects as go`

**Task 2: Load and Inspect the S&P 500 Index Data:**<br>
The S&P 500 Data set is from Kaggle. But I have downloaded the raw files in Github. Accessed via this [**data_link**](https://raw.githubusercontent.com/Lawrence-Krukrubo/Anomaly_Detection_in_Time_Series_Data_with_Keras/master/spx.csv)

**Task 3: Data Preprocessing:**<br>
I shall do some data-preprocessing that will involve:-
* Standardizing the target-vector by removing the mean and scaling it to the unit variance, using the `standard_scaler` function from `sklearn.preprocessing`.
* Then I reshape the Data vectors in the form `n(samples)` by `n(time_steps)` by `n(features)` in line with Time-Series modelling.

**Task 4: Temporalize Data and Create Training and Test Splits:**<br>
Here, I shall create a method that does the following:
* Partition or temporalize the data in a sliding-window of 30 days per partition.
* Save each partition of 30 days as an element in the training vector.
* Add the following day's closing value as a corresponding element in the target vector.
* Repeat these steps for all data values in the S&P data set.
* The method finally returns two arrays of the Training set and the Training labels or Target.
* Finally, I split the data into 80% training and 20% testing sets.

**Task 5: Build an LSTM Autoencoder:**<br>
Here I'd build an LSTM auto-encoder network and visualize its architecture and data flow. Here are the steps:-
* I train an auto-encoder on data with no anomalies.
* Then I take a new data point and try to reconstruct it using the auto-encoder.
* If the reconstruction error for the new data point is above some set threshold, I'd label that data point as an anomaly.

**Task 6: Train the Autoencoder:**<br>
* First I compile the model specifying the loss function to measure and the optimizer function.
* Then, I apply the early stopper regularization technique from `tf.keras.callbacks.EarlyStopping()` function.
* Next I set `validation=0.1` to use 10% of training data for validation.
* I also set other hyper-parameters like `batch-size`, `callbacks` and `shuffle=False` to maintain data order.
* Finally I train the model with a high `epoch`, allowing it to fully train until the early-stopping condition is met.

**Task 7: Plot Metrics and Evaluate the Model:**<br>
Here, I'd use Matplotlib to plot both the training and validation loss. These Loss values are stored in the models's history.history object. So I use this to plot the losses. The history object is juast a Python dictionary.

**Task 8: Detect Anomalies in the S&P 500 Index Data:**
Finally I detect the anomalies, which are the closing values above the set threshold I specified earlier.
* First I build a data frame with the loss and anomaly values(those exceeding the treshold)
* Using this data frame I can identify closing values that are anomalies in the data. and this analysis can be used to classify future closing data.

### Summary:
In this project, I have demonstrated how to combine two very powerful concepts in deep learning:-
* LSTMs
* Auto-Encoders 
And I've combined these two to create an anomaly detection module to predict anomalies in certain price changes in the S&P 500 Index Data.

**Key points:**
The three essential components of an Auto-encoder are: 
* 1. The Encoder. 
* 2. The Decoder.
* 3. the Loss function.

### License:
All files, documents and analysis drawn in this Repo abide under the **MIT license** appended in the root directory of this project.

Finally, I earned a certificate from Coursera for completing this project satisfactorily. See it [**here**](https://coursera.org/share/2f40a928ab2ad329c623dc438d4050ab)


