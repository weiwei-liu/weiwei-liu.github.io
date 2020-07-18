---
layout: post
title: Anomaly Detection with Recurrent Neural Network
subtitle: Anomaly detection in operation trace logger of QNX embedded system products
cover-img: /assets/img/embed.jpg
thumbnail-img: /assets/img/embed.jpg
tags: [Tensorflow, Keras,RNN,anomaly detection, seq2seq attention model]
---

This [notebook](https://github.com/weiwei-liu/anomaly_detection/blob/master/anomaly_detection_NN_train.ipynb) consists of four main parts to describe the workflow of training a Recurrent Neural Network with attention layer model to classify anomaly events in a sequence based embeded software log. These four parts are Data Loading and Preprocessing, Model building, Model training, Results Analysis and Visualization.

The evaluate of new test data will be presented in [another notebook](https://github.com/weiwei-liu/anomaly_detection/blob/master/anomaly_detection_NN_test.ipynb).

Check the [repository](https://github.com/weiwei-liu/anomaly_detection) for code details.


#### Data loading and preprocessing

* Load in all 15 .csv data files, and save as pandas dataframes.


#### Overview of the data

* Groupby `class` and `event` column in the dataframe to get the occurrence count of different events under different class.


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>clean1</th>
      <th>clean2</th>
      <th>anomalous1</th>
      <th>anomalous2</th>
    </tr>
    <tr>
      <th>class</th>
      <th>event</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="9" valign="top">COMM</th>
      <th>MSG_ERROR</th>
      <td>6.0</td>
      <td>6.0</td>
      <td>5715</td>
      <td>422.0</td>
    </tr>
    <tr>
      <th>REC_MESSAGE</th>
      <td>17968.0</td>
      <td>17969.0</td>
      <td>65232</td>
      <td>45072.0</td>
    </tr>
    <tr>
      <th>REC_PULSE</th>
      <td>24710.0</td>
      <td>24226.0</td>
      <td>28312</td>
      <td>39529.0</td>
    </tr>
    <tr>
      <th>REPLY_MESSAGE</th>
      <td>17947.0</td>
      <td>17950.0</td>
      <td>59477</td>
      <td>44565.0</td>
    </tr>
    <tr>
      <th>SIGNAL</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>37</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>SND_MESSAGE</th>
      <td>18089.0</td>
      <td>18077.0</td>
      <td>65378</td>
      <td>45426.0</td>
    </tr>
    <tr>
      <th>SND_PULSE</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>882</td>
      <td>11289.0</td>
    </tr>
    <tr>
      <th>SND_PULSE_DIS</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>877</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SND_PULSE_EXE</th>
      <td>36701.0</td>
      <td>48219.0</td>
      <td>181312</td>
      <td>160365.0</td>
    </tr>
    <tr>
      <th>CONTROL</th>
      <th>BUFFER</th>
      <td>2161.0</td>
      <td>2158.0</td>
      <td>4845</td>
      <td>4334.0</td>
    </tr>
  </tbody>
</table>
</div>



From above table, it could be seen that the clean and anomalous files are quite different based on the occurrence counts of different events. For example, normally event `COMM-SND_MESSAGE` occurced around 18000 times, while in the anomalous files it occured around 45000~67000 times. This may not be seen as an effective way to detect anomalous activity, however, it can show a general picture of the data where the anomaly could be residing.

Before the data is ready for use in the training period. The data was tokenized, preprocessed to input sequences and output sequences, and then split into training and validation dataset.


#### Model Building

In this project, sequence to sequence neural network model with attention layer was used. The seq2seq model contains an encoder with GRU unit, an attention layer which increase the accuracy of the model, and a decoder also with GRU unit.

The architecture of this model is:

<div style="text-align:center">
<img src="/assets/img/attention.png" alt="seq2seq model architecture" width="250" height="350"/>
</div>

The input is a small segment of the log file, in this case, 5 continuous events, and the target output is the next 5 continuous events following the input one. The general idea is that using this proposed NN model to train inputs and predicting the following outputs. Assuming the event sequences patterns between the clean and anomalous ones are different, then the preciting/test accuracy should be different using the same model and trained weights.

Check [model.py](https://github.com/weiwei-liu/anomaly_detection/blob/master/model.py) file for the details of encoder, attention, and decoder models.

#### Model Training

Check [training code](https://github.com/weiwei-liu/anomaly_detection/blob/master/anomaly_detection_NN_train.ipynb) for details.

#### Results

The next step is to predict results using the above model and trained weights of each layer.

The test inputs are processed using event sequence length `Tx = 5`, same as the trained data, while using stride `stride = 5` instead of 2.

Save all the predicted result into .npy files for further analysis use.

* Set anomaly creteria

As mentioned above, Assuming the event sequences patterns between the clean and anomalous ones are different, then the preciting/test accuracy should be different using the same model and trained weights.

In the following code, I use squence length of 1000 as one input sample, and use the above trained model to precited output, and then compare the precited output with target values to get the misclassification accuracy.


After predict outputs on all the 10 clean files, calculate the mean and variance of the misclassification accuracy. Finally, I set the criteria to be (mean + 3* standard_deviation).

Any 1000 events long sequence with  misclassification rate higher than the criteria will be deemed as anomaly segment.


**In this case, any misclaasification rate higher than 0.365 will be classified as anomaly event.**

* Visualize anomalous events

**Normal sequences**



![png](/assets/img/output_49_1.png)


**Abnormal sequences A**



![png](/assets/img/output_49_21.png)

**Abnormal sequences B**


![png](/assets/img/output_49_29.png)




#### Reference

* O. M. Ezeme, Q. H. Mahmoud and A. Azim, "DReAM: Deep Recursive Attentive Model for Anomaly Detection in Kernel Events," in IEEE Access, vol. 7, pp. 18860-18870, 2019.

* https://www.tensorflow.org/tutorials/text/nmt_with_attention
