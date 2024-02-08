# Electricity Theft Detection
There are two types of losses that a power system faces, one is technical losses (TL) and the other is non-technical losses (NTL). TL are unavoidable due to their inherent nature in transformers, cables, and long-distance lines during energy transfer. NTL has consistently troubled utility companies, primarily stemming from two main sources: electricity theft and the failure to pay utility bills. Every year electricity theft costs loss billions of dollars to utility companies. 

To overcome this theft issue, a reliable system is required that detects electricity theft accurately and is easy to deploy. This task is handled using machine learning and deep learning techniques. These techniques are most dependable in detecting electricity theft because they can easily learn the complex patterns of the theft users, and the trained system is easily deployable. The following techniques will be used in this project,
1. Logistic Regression - Random Forest Stacking Model (LR-RF)
2. Multilayer Perceptron (MLP)
3. Recurrent Neural Network (RNN)

## Models Overview
### Logistic Regression - Random Forest Stacking Model (LR-RF)
The LR-RF stacking model combines the simplicity and interpretability of Logistic Regression with the robustness and flexibility of Random Forest. Stacking these models allows for the integration of their predictive capabilities, where the Random Forest and Logistic Regression both serve as the base models to capture complex patterns and non-linear relationships, and the Logistic Regression acts as the meta-model to consolidate the predictions into a final verdict. The motivation behind using the LR-RF stacking model is to leverage the complementary strengths of both algorithms. Logistic Regression, while linear, is effective at providing probabilistic outcomes, making it easy to interpret results. On the other hand, Random Forest excels in handling high-dimensional data and modeling complex interactions between variables. By stacking them, we aim to enhance the model's accuracy and generalizability, making it adept at distinguishing between normal and theft-related consumption patterns.

### Multilayer Perceptron (MLP)
The MLP is a class of feedforward artificial neural networks that consists of at least three layers of nodes: an input layer, hidden layers, and an output layer. MLP utilizes a backpropagation technique for training, enabling it to capture deep non-linear relationships in the data. The choice of MLP for electricity theft detection is driven by its capacity to model complex and high-dimensional datasets. Its deep learning architecture can learn features automatically, reducing the need for manual feature engineering. This makes MLP particularly useful for analyzing the intricate and subtle variances in electricity usage data that may indicate theft, thereby improving detection accuracy.

### Recurrent Neural Network (RNN)
RNNs are a class of neural networks that are especially powerful for modeling sequential data, thanks to their internal state (memory) that captures information about sequences they've seen so far. This makes them ideal for time-series data like electricity consumption records. The motivation for using RNNs lies in their ability to process time-series data and recognize temporal patterns, which is essential for analyzing electricity consumption over time. Electricity theft often exhibits distinct temporal patterns that may not be evident in cross-sectional data analysis. RNNs can identify these patterns, making them a valuable tool for detecting irregularities in consumption that could signify fraudulent activities.

## Installation

To run the models and evaluate their performance, you need to install the following Python libraries.
1. **scikit-learn**: For Logistic Regression, Random Forest, and various metrics like confusion matrix and classification report.
2. **TensorFlow or Keras**: For building and training MLP and RNN models.
3. **matplotlib and seaborn**: For plotting training and testing accuracy and other visualizations.
4. **pandas**: For data manipulation and analysis.
5. **numpy**: For numerical computations.

You can install them using pip, Python's package installer. Open your terminal or command prompt and run the following commands:

```bash
pip install scikit-learn tensorflow keras matplotlib seaborn pandas numpy
```
## Dataset Overview
A publicly available dataset from SGCC (State Grid Corporation of China) is used for the training of models. This dataset is curated to support the development and evaluation of machine-learning models aimed at detecting electricity theft. It comprises various features indicative of consumer electricity usage patterns, potentially flagged for irregularities that may suggest theft.

### Features Description
The dataset likely includes several columns such as:

**Customer ID:** Unique identifier for each customer (e.g., CONS_NO).
**Date/Time:** Timestamps representing the period of electricity consumption.
**Electricity Usage:** Daily electricity usage metrics.
**Flag:** Indicates whether electricity theft is suspected (e.g., 1 for theft, 0 for no theft).

### Target Variable
The target variable is designed to indicate instances of electricity theft. 
- flag  '1' represents suspected theft
- flag '0' indicates normal usage

## Data Pre-Processing
In the process of preparing our dataset for the electricity theft detection project, we encountered several data-related challenges that could potentially impact the performance of our machine-learning models. Firstly, the dataset had missing values, which could introduce bias or inaccuracies in the model predictions. Secondly, the presence of outliers in the data could skew our analysis, leading to misleading results. Lastly, the issue of class imbalance was evident, with a significant disproportion between the classes of interest, which could bias the model towards the majority class, diminishing its ability to detect the minority class accurately.

To address these challenges, we implemented the following solutions:

- **For Missing Values**:
  - KNN Imputer (k=20)
- **For Outliers:**
  - Local Outlier Factor (LOF)
- **For Class Imbalance:**
  - ADASYN (to oversample the minority class)
  - Near Miss (to undersample the majority class)

### KNN Imputer 
Missing values in the dataset can significantly impact the performance of machine learning models. We employed the KNN (K-Nearest Neighbors) imputer with k=20 to estimate and fill in these missing values. This method was chosen because it predicts the missing value based on the 20 nearest neighbors, providing a more nuanced and accurate imputation than simpler methods like mean or median imputation. It leverages the underlying patterns in the data, ensuring that the imputed values are realistic and consistent with the surrounding data points.
```bash
# Select columns for KNN imputation (excluding the label column)
knn_imputation_data = data.iloc[:, 1:]

# Create a KNNImputer instance
knn_imputer = KNNImputer(n_neighbors=20)

# Apply KNN imputation along axis=1
imputed_data = knn_imputer.fit_transform(knn_imputation_data.T).T
```
### Local Outlier Factor (LOF)
The presence of outliers can skew the model training process, leading to poor generalization on unseen data. The LOF algorithm was utilized to identify and handle outliers in the dataset. By measuring the local density deviation of a given data point with respect to its neighbors, LOF helps in pinpointing observations that significantly differ from the norm. This approach is particularly effective for datasets where the definition of an "outlier" is not absolute but contextually dependent on the local data structure.
```bash
from sklearn.neighbors import LocalOutlierFactor

# Create an instance of the Local Outlier Factor model
lof = LocalOutlierFactor(n_neighbors=10, contamination='auto')

# Fit the LOF model to the data and predict outliers
outlier_predictions = lof.fit_predict(imputed_data_10)

# Convert the outlier predictions to a boolean mask
outlier_mask = outlier_predictions == -1

# Calculate the median values for each column in imputed_data_10
median_values = imputed_data_10.median()

# Replace outliers with the median values
imputed_data_10[outlier_mask] = median_values

# Now, imputed_data_10 contains outliers replaced with median value
LOF_DATA = imputed_data_10
```


### ADASYN (Adaptive Synthetic Sampling)
Class imbalance is a common issue in datasets, especially in contexts like electricity theft detection, where instances of theft may be much rarer than normal usage patterns. ADASYN was used to address this by generating synthetic samples of the minority class. It adaptively adjusts the number of synthetic samples according to the learning 
difficulty of the minority class, thus improving model robustness and performance by providing a more balanced class distribution for training.
```bash
from imblearn.over_sampling import ADASYN

# Convert imputed data to a DataFrame and prepare X, y
imputed_data_10 = pd.DataFrame(imputed_data_10)
X = imputed_data_10.iloc[:, 1:]  # Features
y = data['FLAG']  # Labels

# Apply ADASYN
adasyn = ADASYN()
X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
```


### Near Miss
As another technique to counteract class imbalance, Near Miss is an undersampling approach that selects samples from the majority class based on their distance to the minority class. This ensures that the remaining samples in the majority class are the most relevant for the model to learn the distinguishing characteristics of each class. By reducing the size of the majority class, Near Miss helps in equalizing the class distribution, facilitating a more effective learning process.
```bash
from imblearn.under_sampling import NearMiss

# Apply NearMiss undersampling
n_neighbors = min(sum(y_adasyn == 1), 3)  # Limit n_neighbors to the number of minority samples
if n_neighbors == 0:
    n_neighbors = 1  # Set a default value of 1 if there are no minority samples
nm = NearMiss(sampling_strategy='auto', n_neighbors=n_neighbors, version=1)
X_final, y_final = nm.fit_resample(X_adasyn, y_adasyn)
```
## Splitting Data for Testing and Training
In our project's data preparation phase, we utilized the `train_test_split` function from Scikit-learn's `model_selection` module to divide our dataset into training and testing sets. This crucial step allows us to train our machine learning models on a subset of the data (training set) and then evaluate their performance on unseen data (testing set). We specified a test size of 20% (`test_size=0.2`), meaning that 20% of the data is reserved for testing, while the remaining 80% is used for training. Additionally, we set a `random_state` of 42 to ensure reproducibility of our results. The variables `X_final` and `y_final` represent the processed features and target variable, respectively, ready to be used in training and evaluating our electricity theft detection models.
```bash
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
```
## Model Training
Each model was trained individually on the training dataset and eventually tested with the new unseen testing data, here's how it's done,

### Logistic Regression - Random Forest Stacking Model:

In our electricity theft detection project, we've implemented a sophisticated modeling technique known as stacking to enhance the predictive performance by combining the strengths of Logistic Regression and Random Forest Classifier. This approach involves layering or 'stacking' multiple base models with a meta-model on top, which learns to optimally combine the predictions of the base models to make a final prediction.

**Base Models**: We selected Logistic Regression and Random Forest Classifier as our base models due to their complementary nature - Logistic Regression for its simplicity and efficiency in linearly separable data, and Random Forest for its ability to handle non-linear data with a complex structure.

**Meta-Model:** A Logistic Regression model serves as the meta-model, which integrates the predictions from the base models to make the final prediction, leveraging the diversity of the base models to improve overall accuracy.

**Training:** The stacking model is trained on the dataset split into training and testing sets, with 80% of the data used for training and 20% reserved for testing, ensuring a balanced approach to learning and validation.

```bash
#Stacking
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier


# Define the base models
base_models = [
    ('logreg', LogisticRegression()),
    ('rf', RandomForestClassifier())
]

# Define the meta-model
meta_model = LogisticRegression()

# Create the stacking model
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Train the stacking model on the training data
stacking_model.fit(X_train, y_train)

# Make predictions using the stacking model
predictions = stacking_model.predict(X_test)
```

### Multilayer Perceptron :
We have constructed a neural network model to address the challenge of classifying electricity usage patterns. The process involves several critical steps, from data splitting and label encoding to hyperparameter optimization and model training, each tailored to enhance the model's predictive accuracy.

**Data Preparation and Encoding:** Initially, we divided our dataset into training and testing subsets, ensuring a comprehensive evaluation framework. A test size of 20% guarantees a balanced split, providing ample data for both training and validation phases. Following the split, we faced the task of preparing our labels for the neural network. Given our binary classification goal, encoding the labels into a binary class matrix was imperative. This conversion, facilitated by the to_categorical function, transforms our labels into a format compatible with the softmax activation function used in the neural network's output layer, catering to our two-class problem.

**Hyperparameter Tuning:** A pivotal aspect of our methodology was the optimization of hyperparameters, a process that led us to identify an optimal configuration: 192 units for our neural network's layers and a learning rate of 0.0001. This optimization ensures that our model's architecture and learning pace are finely tuned to the intricacies of our dataset, significantly contributing to the model's efficacy.

**Neural Network Architecture:** Leveraging the insights gained from hyperparameter optimization, we constructed our neural network. The model features a dense layer with 192 units, employing a relu activation function for non-linearity, followed by an output layer with a softmax activation tailored for multi-class classification. This architecture is designed to capture the complex relationships within our data, facilitating accurate predictions of electricity theft.

**Training and Evaluation:** The training process was meticulously planned, with the model being fed the standardized training data and the corresponding encoded labels. Over 50 epochs, the model learned to distinguish between normal and fraudulent electricity usage patterns, with the validation data providing a benchmark for its performance. This approach not only ensures the model's robustness but also its ability to generalize well to new, unseen data.

```

num_classes = 2  #  number of classes

# Data Splitting  into training and testing  data sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# y_final is your original labels
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_test_encoded = to_categorical(y_test, num_classes=num_classes)

# Best Hyperparameters through (Hyper parameter Optimization)
best_hyperparameters = {'units': 192, 'learning_rate': 0.0001}
#neural network model
model = Sequential()
model.add(Dense(units=best_hyperparameters['units'], activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(learning_rate=best_hyperparameters['learning_rate'])
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# model Training
model.fit(X_train, y_train_encoded, epochs=50, validation_data=(X_test, y_test_encoded))
```




### Recurrent Neural Network :
we've taken an advanced approach by incorporating Recurrent Neural Networks (RNN), specifically utilizing Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) layers, to model the sequential nature of electricity consumption data. This method allows us to capture temporal dependencies and patterns that are crucial for identifying irregularities indicative of theft.

**Preprocessing:** The input features are standardized using the StandardScaler to ensure that the neural network receives data within a scale conducive to effective learning. Given the sequential input required by RNNs, we reshape our data into a format suitable for LSTM layers, adjusting the data into sequences of a specified timestep.

**Model Construction:** Our neural network model is built with a Sequential configuration, starting with an LSTM layer designed to process the sequential input, followed by layers aimed at enhancing model learning and preventing overfitting:

**LSTM Layer:** Captures temporal dependencies with 128 units, returning sequences to allow stacking of LSTM layers or subsequent processing.
**LeakyReLU Activation:** Provides non-linearity, allowing the model to learn complex patterns.
**BatchNormalization:** Normalizes the activations from the previous layer, improving stability and speed of training.
**Dropout:** Regularization technique to prevent overfitting by randomly setting input units to 0 during training with a rate of 0.3.
**Dense Layers:** Fully connected layers with regularization (L2) and dropout to consolidate learned features into predictions.
**Output Layer:** A single unit with a sigmoid activation function to output a probability, indicating the likelihood of electricity theft.
**Compilation and Training:** The model is compiled with the Adam optimizer and binary cross-entropy loss, reflecting our binary classification goal. Training occurs over 50 epochs with a batch size of 128, including a validation split to monitor and mitigate overfitting.
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


# Assuming X_train_scaled, X_test_scaled, y_train, y_test are already defined and properly sequenced

# Standardize the input features for the neural network
scaler = StandardScaler()
timesteps = 1  # Adjust this based on your specific data

# Function to reshape the data into (samples, timesteps, features)
def create_sequences(data, timesteps):
    X = []
    for i in range(len(data) - timesteps + 1):
        X.append(data[i:(i + timesteps)])
    return np.array(X)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for LSTM layer

X_train_scaled_reshaped = create_sequences(X_train_scaled, timesteps)
X_test_scaled_reshaped = create_sequences(X_test_scaled, timesteps)
# Adjust y_test to match the number of sequences in X_test_scaled_reshaped

y_test_adjusted = y_test[timesteps - 1:]
y_train_adjusted = y_train[timesteps - 1:]


nn_model_rnn = Sequential([
    LSTM(128, input_shape=(timesteps, X_train_scaled_reshaped.shape[2]), return_sequences=True),
    LeakyReLU(),
    BatchNormalization(),
    Dropout(0.3),


    Dense(64, kernel_regularizer=l2(0.001)),
    LeakyReLU(),
    Dropout(0.3),


    Dense(32, kernel_regularizer=l2(0.001)),
    LeakyReLU(),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

nn_model_rnn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = nn_model_rnn.fit(X_train_scaled_reshaped, y_train_adjusted, epochs=50, batch_size=128, validation_split=0.2)
```
## Results and Observations
We evaluate each model's performance on the testing set using a classification report that provides detailed metrics like precision, recall, and F1-score. Additionally, we assess the model's ability to distinguish between classes by calculating the AUC (Area Under the Curve) from the ROC (Receiver Operating Characteristic) curve, alongside plotting the curve to visualize performance. To further analyze the model's predictive capability, we generate a confusion matrix visualized as a heatmap. This not only shows the number of correct and incorrect predictions but also offers insights into the type of errors made by the model.

### Classification Reports:
A comprehensive table that summarizes the classification reports for each model is shown below. This table will encapsulate key performance metrics such as precision, recall, F1-score, and accuracy, providing a clear, concise comparison of each model's effectiveness in detecting electricity theft.

$$
\begin{array}{|c|c|c|c|c|}
\hline
\textbf{Model} & \textbf{Recall} & \textbf{Precision} & \textbf{Accuracy} & \textbf{F1 Score} \\
\hline
LR-RF Stack & 0.95 & 0.95 & 0.95 & 0.95 \\
\hline
RNN & 0.92 & 0.92 & 0.92 & 0.92 \\
\hline
MLP & 0.91 & 0.91 & 0.91 & 0.91 \\
\hline
\end{array}$$

### Comparative Analysis of ROC Curve:
The graph shown below presents a comparison of Receiver Operating Characteristic (ROC) curves for three different models: Logistic Regression - Random Forest Stacking (LR-RF), Recurrent Neural Network (RNN), and Multilayer Perceptron (MLP). ROC curves are a graphical representation of a classification model's diagnostic ability as its discrimination threshold is varied. The True Positive Rate (TPR, or sensitivity) is plotted against the False Positive Rate (FPR, or 1 - specificity) at various threshold settings. The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test. The diagonal dashed line represents a random guess, and the ideal point is the top left corner of the graph, indicating a high true positive rate and a low false positive rate. The curves of all three models are clustered closely together and approach the ideal point, which suggests that each model exhibits a strong ability to discriminate between the positive and negative classes. The overlapping nature of the curves indicates that the performance of the three models is quite similar in terms of ROC AUC metrics.

![AUC](https://github.com/shahid-ali2134/ElectricityTheftDetection/assets/88580273/3425ad7e-8c4d-4c83-a8d8-8dfa4d5c780b)


### Confusion Matrices:

**Logistic Regression - Random Forest Stacking Model:**

![stack1](https://github.com/shahid-ali2134/ElectricityTheftDetection/assets/88580273/54238f31-6a80-4784-abdc-8f9dab2ee1f3)


**Multilayer Perceptron:**

![mlp](https://github.com/shahid-ali2134/ElectricityTheftDetection/assets/88580273/fb88ed7d-80bf-4f1b-bd01-c850f3d4be06)


**Recurrent Neural Network:**

![rnn1](https://github.com/shahid-ali2134/ElectricityTheftDetection/assets/88580273/32fa99cd-68f0-4283-bc60-3d0536fed04c)

## Conclusion
This study explored the energy usage patterns of irregular consumers to identify electricity theft effectively. Tackling the prominent issue of class imbalance within the dataset, we initiated with comprehensive data preprocessing to mitigate this challenge, followed by the deployment of various classification models. The LR-RF Stacking Model, RNN, and MLP were employed, each demonstrating efficacy in discerning between fraudulent and non-fraudulent users. Our real-world application yielded notable results, with the LR-RF model achieving high scores across accuracy, precision, recall, and F1-score, demonstrating exceptional capability in identifying theft while minimizing false positives. The RNN model, while slightly less accurate overall, showed consistent performance, proving particularly adept in handling sequential data. The MLP offered a more straightforward yet effective alternative, suited for less complex scenarios with an accuracy of 0.91. These findings underscore the importance of model selection tailored to the specific nature of the data and the requirements of the application, where LR-RF may excel in general, but RNN's sequential data processing can be advantageous in certain contexts, and MLP's simplicity benefits processing efficiency.

## References
- Z. Zheng, Y. Yang, X. Niu, H. Dai, Y. Zhou, "Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids," *IEEE Transactions on Industrial Informatics*, vol. 14, no. 4, pp. 1606-1615, April 2018. [DOI: 10.1109/TII.2017.2785963]([https://doi.org/10.1109/TII.2017.2785963](https://ieeexplore.ieee.org/document/8233155/)https://ieeexplore.ieee.org/document/8233155/)
- Inam Ullah Khan, Nadeem Javeid, C. James Taylor, and Xiandong Ma, "Robust Data Driven Analysis for Electricity Theft Attack-Resilient Power Grid," *Lancaster University, Lancaster, UK, 2022.*
