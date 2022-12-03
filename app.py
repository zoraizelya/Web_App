# Libraries
import streamlit as st  # for Web App
import numpy as np # for data manipulation
import matplotlib.pyplot as plt # for data visualization
from sklearn import datasets # for data
from sklearn.model_selection import train_test_split # for data splitting
from sklearn.decomposition import PCA # for dimensionality reduction
from sklearn.svm import SVC # for classification
from sklearn.neighbors import KNeighborsClassifier # for classification
from sklearn.ensemble import RandomForestClassifier # for classification
from sklearn.metrics import accuracy_score # for model evaluation

# Title / Heading

st.write("""
# Lets Explore Different ML Models And Datasets
lets see which of them are good for which dataset?
""")

# Sidebar
dataset_name = st.sidebar.selectbox ("Select Dataset",
                                    ("Iris", "Breast Cancer", "Wine Dataset")
                                    )
classifier_name = st.sidebar.selectbox ("Select Classifier",
                                       ('KNN', 'SVM', 'Random Forest')
                                       )
# Function to get dataset
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

# Function to add parameters
X, y = get_dataset(dataset_name)

# shape of dataset
st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))

# Classifiers Parameters
def add_parameter_ui(classifier_name):
    params = dict() # empty dictionary
    if classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K # K is the number of neighbors
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C # C is the regularization parameter
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth # max_depth is the maximum depth of the tree
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators # n_estimators is the number of trees in the forest
    return params
# Function to get classifier
params = add_parameter_ui(classifier_name)

# get classifier
def get_classifier(classifier_name, params):
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], random_state=1234)
    return clf
# get classifier results
clf = get_classifier(classifier_name, params)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# training the classifier
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

# Accuracy Score Check
acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

# Plotting the graph
pca = PCA(2) # 2 components
X_projected = pca.fit_transform(X)
# Dimentions of the dataset
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure() # empty figure
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis") # scatter plot
plt.xlabel("Principal Component 1") # x-axis label
plt.ylabel("Principal Component 2") # y-axis label
plt.colorbar() # color bar

# show the plot
st.pyplot(fig)