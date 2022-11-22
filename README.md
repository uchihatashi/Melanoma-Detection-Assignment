# Melanoma-Detection-Assignment
### Problem Statement
In this assignment, you will build a **multiclass classification model using a custom convolutional neural network in TensorFlow.** 

 

**Problem statement:** To build a CNN based model which can accurately detect **melanoma**. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.


You can download the dataset [here](https://drive.google.com/file/d/1xLfSQUGDl8ezNNbUkpuHOYvSpTyxVhCs/view?usp=sharing)


The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.


The data set contains the following diseases:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

NOTE:

* You don't have to use any pre-trained model using Transfer learning. All the model building processes should be based on a custom model.
* Some of the elements introduced in the assignment are new, but proper steps have been taken to ensure smooth learning. You must learn from the base code provided and implement the same for your problem statement.
* The model training may take time to train as you will be working with large epochs. It is advised to use GPU runtime in [Google Colab](https://colab.research.google.com/).
 

## Project Pipeline
- **Data Reading/Data Understanding** → Defining the path for train and test images 
- **Dataset Creation** → Create train & validation dataset from the train directory with a batch size of 32. Also, make sure you resize your images to 180*180.
- **Dataset visualisation** → Create a code to visualize one instance of all the nine classes present in the dataset 
- **Model Building & training :**
    - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
    - Choose an appropriate optimiser and loss function for model training
    - Train the model for ~20 epochs
    - Write your findings after the model fit. You must check if there is any evidence of model overfit or underfit.
- **Chose an appropriate data augmentation strategy to resolve underfitting/overfitting**
- **Model Building & training on the augmented data:**
    - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model rescale images to normalize pixel values between (0,1).
    - Choose an appropriate optimiser and loss function for model training
    - Train the model for ~20 epochs
    - Write your findings after the model fit, see if the earlier issue is resolved or not?
- **Class distribution:** Examine the current class distribution in the training dataset 
    - Which class has the least number of samples?
    - Which classes dominate the data in terms of the proportionate number of samples?
- **Handling class imbalances:** Rectify class imbalances present in the training dataset with [Augmentor](https://augmentor.readthedocs.io/en/master/) library.
- **Model Building & training on the rectified class imbalance data :**
    - Create a CNN model, which can accurately detect 9 classes present in the dataset. While building the model, rescale images to normalize pixel values between (0,1).
    - Choose an appropriate optimiser and loss function for model training
    - Train the model for ~30 epochs
    - Write your findings after the model fit, see if the issues are resolved or not?


### Conclusion:

After applying all Dropout, BatchNormalization(), Augmenting data, Handling class imbalances we got this as result:

**Final model accuracy:**

- The final train accuracy is 0.9343
- Validation accuracy is 0.8567
- The test accuracy is 0.4746

#### **We can still improve the accuracy, especially for test dataset, but we need more data. As of now, we have really fewer data, and even we can augment more data. Here we created only 500 images for each class; in the future, we can create more like 1.5k+.**

But if we use more image, it will take more time and computation power.

#### **We can still improve the accuracy, especially for test dataset, but we need more data. As of now, we have really fewer data, and even we can augment more data. Here we created only 500 images for each class; in the future, we can create more like 1.5k+.**

But if we use more image, it will take more time and computation power.



### Conclusion:

**Here are all the models with their respective accuracy and number of parameters**

| Model No. | Model Type      | No. parameters | Valadation Acc(%) | Tranning Acc(%) | Input Parameter                                                                                         |
| --------- | --------------- | -------------- | ----------------- | --------------- | ------------------------------------------------------------------------------------------------------- |
| 0         | 3D CNN          | 9,00,805       | 19                | 88.84           | HxW 120, frame=30, batch\_size=55, epoch=20                                                             |
| 1         | 3D CNN          | 19,67,813      | 22                | 74.81           | HxW 120, frame=30, batch\_size=55, epoch=20                                                             |
| 2         | 3D CNN          | 19,67,813      | 21                | 92.91           | HxW 120, frame=30, batch\_size=55, epoch=15, dropout=0.5                                                |
| 3         | 3D CNN          | 17,62,613      | 68                | 83.26           | HxW 120, frame=30, batch\_size=55, epoch=25, dropout=0.5, filter=2                                      |
| 4         | 3D CNN          | 25,56,533      | 73                | 91.86           | HxW 120, frame=20, batch\_size=30, epoch=25, filter=3, dense\_neurons=256                               |
| 5         | 3D CNN          | 25,56,533      | 21                | 48.72           | HxW 120, frame=20, batch\_size=30, epoch=25, dropout=0.5, filter=3, dense\_neurons=256                  |
| 5.1       | 3D CNN          | 25,56,533      | 25                | 85.67           | HxW 120, frame=20, batch\_size=30, epoch=25, dropout=0.25, filter=3, dense\_neurons=256                 |
| 6         | 3D CNN          | 9,08,645       | 74                | 88.99           | HxW 120, frame=16, batch\_size=20, epoch=20, dropout=0.25, filter=2, dense\_neurons=128                 |
| 7         | 3D CNN          | 4,94,981       | 74                | 82.35           | HxW 120, frame=16, batch\_size=20, epoch=20, dropout=0.25, filter=2, dense\_neurons=64                  |
| 8         | CNN-LSTM        | 16,56,453      | 81                | 88.39           | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=3, dense\_neurons=128, lstm\_cell=128 |
| 8.1       | CNN-GRU         | 1,346,405      | 75                | 93.21           | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=3, dense\_neurons=128, GRU\_cell=128  |
| 9         | 3D CNN          | 19,66,309      | 79                | 71.87           | HxW 120, frame=20, batch\_size=20, epoch=25, dropout=0.5, filter=3, dense\_neurons=256                  |
| 10        | 3D CNN          | 17,61,109      | 48                | 68.7            | HxW 120, frame=16, batch\_size=30, epoch=25, dropout=0.5, filter=2, dense\_neurons=256                  |
| 11        | 3D CNN          | 25,54,549      | 83                | 70.21           | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.5, filter=2, dense\_neurons=256                  |
| 12        | 3D CNN          | 25,54,549      | 78                | 92.76           | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=2, dense\_neurons=256                 |
| 13        | 3D CNN          | 9,09,637       | 88                | 92.38           | HxW 120, frame=30, batch\_size=20, epoch=25, dropout=0.25, filter=3,2, dense\_neurons=128               |
| 13.1      | 3D CNN          | 9,07,733       | 85                | 91.1            | HxW 120, frame=16, batch\_size=20, epoch=35, dropout=0.25, filter=2, dense\_neurons=128                 |
| **13.2**  | **3D CNN**      | **9,09,637**   | **94**            | **91.7**        | **HxW 120, frame=16, batch\_size=20, epoch=35, dropout=0.25, filter=3,2, dense\_neurons=128**           |
| 13.3      | 3D CNN          | 9,09,637       | 66                | 93.06           | HxW 120, frame=16, batch\_size=32, epoch=35, dropout=0.25, filter=3,2, dense\_neurons=128               |
| 14        | 3D CNN          | 4,94,245.00    | 81                | 85.9            | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=2, dense\_neurons=64                  |
| 15        | CNN-GRU         | 25,57,413      | 82                | 99.55           | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=3, dense\_neurons=128, GRU\_cell=128  |
| 15.1      | CNN-LSTM        | 33,76,357      | 78                | 99.4            | HxW 120, frame=16, batch\_size=20, epoch=25, dropout=0.25, filter=3, dense\_neurons=128, LSTM\_cell=128 |
| 16        | TL-LSTM         | 3,840,453      | 72                | 99.55           | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=128, LSTM\_cell=128 |
| 16.1      | TL-LSTM         | 35,16,229      | 77                | 99.1            | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=64, LSTM\_cell=64   |
| 17        | TL-GRU          | 36,93,253      | 74                | 99.85           | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=128, GRU\_cell=128  |
| 18        | TL-GRU          | 34,46,725      | 70                | 99.4            | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=64, GRU\_cell=64    |
| 19        | TL-GRU-Non\_AUG | 3,446,725      | 71                | 99.25           | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=64, GRU\_cell=64    |
| 20        | TL-GRU-B2       | 83,81,950      | 16                | 20              | HxW 120, frame=16, batch\_size=10, epoch=20, dropout=0.25, filter=3, dense\_neurons=128, GRU\_cell=128  |


### So, here we have concluded that **Model no 13.2** gave the best accuracy score both for Validation(94%) and training (91.7%).
