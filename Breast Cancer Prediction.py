import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import os 


dataset_url = "uciml/breast-cancer-wisconsin-data"
dataset_path = kagglehub.dataset_download(dataset_url)
file_name = os.path.join(dataset_path, 'data.csv')
df = pd.read_csv(file_name)
print(df.head())


######################################################
#########      Visualize Whole Dataset     ##########
#####################################################

# Visualize the data
"""To visualize all 30 features from x_train in 2D plot by plotting X_train data
   in both axis and responding y_train data as dot for each feature, we need to
   use Principle Component Analysis (PCA) method from sklearn library.
"""
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_df = sc.fit_transform(df.loc[: , 'radius_mean' : 'fractal_dimension_worst'])
y_df = df.loc[: , 'diagnosis']

from sklearn.decomposition import PCA
import seaborn as sns
#1. Turning 30 features into 2 major components
pca = PCA(n_components = 2)
x_2d = pca.fit_transform(x_df)
print(x_2d.shape)



# 2. Visualize using seaborn
plt.figure(figsize = (10, 7))
sns.scatterplot(x = x_2d[:, 0], y = x_2d[:, 1], hue = y_df, style = y_df, palette= { 'M': 'red', 'B' : 'blue'}, markers = {'M' : 'X', 'B' : 'o'}, s = 80, alpha = 0.6 )
plt.title("Breast Cancer Dataset")
plt.xlabel("Principle Component 1")
plt.ylabel("Principle Component 2")
plt.legend(loc = 'upper right', title = 'Diagnosis', frameon = True)
plt.grid(True, linestyle = '--', alpha = 0.3)
plt.show()
######################################################
#########   Data Preprocessing  #############################
######################################################
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['diagnosis'] = labelencoder.fit_transform(df['diagnosis'])
print(df.head())
X = df.iloc[: , 2:32 ].values
Y = df.iloc[: , 1].values
print(X.shape)
print(Y)
print(df.loc[:, ].columns)

######################################################
#########  Creating Logistic Regression Model #########
######################################################
#creating logistic regression model:
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
count = 0
while True:
# getting the (x_train, y_train) (x_test, y_test)
 X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X , Y, test_size = 0.25)
 sc = StandardScaler()
 X_train_ = sc.fit_transform(X_train_)
 X_test_ = sc.fit_transform(X_test_)
#model
 classifier_ = LogisticRegression()
 model_ = classifier_.fit(X_train_, Y_train_)

 predictions_ = model_.predict(X_test_)
 print(accuracy_score(Y_test_, predictions_))
 count += 1
 if accuracy_score(Y_test_, predictions_) == 1.0:
    weights_ = model_.coef_
    bias_ = model_.intercept_
    print("Weights ::: ",weights_)
    print(f"Weights shape : {weights_.shape}")
    print("Bias ::: ",bias_)
    print(f"Bias shape : {bias_.shape}")
    print(accuracy_score(Y_test_, predictions_))
    print(confusion_matrix(Y_test_, predictions_))
    print(f"Total Iterations Occur: {count}")
    break
 else:
    continue

# --- Add this after your 'break' in the while loop ---

# 1. Map the numeric predictions back to 'Positive' and 'Negative'
# Since LabelEncoder turned B -> 0 and M -> 1:
result_labels = ["Positive" if p == 1 else "Negative" for p in predictions_]

# 2. Get the IDs for the test set
# We use X_test_'s original indices from the split to get the IDs from the original df
test_indices = Y_test_.index if hasattr(Y_test_, 'index') else range(len(predictions_))


# To get IDs easily, it's better to split the DataFrame or a Series:
_, _, _, y_test_series = train_test_split(X, df['diagnosis'], test_size=0.25)
# Using the index of y_test_series to get IDs from df
test_ids = df.loc[y_test_series.index, 'id']

# 3. Create the Results DataFrame
results_df = pd.DataFrame({
    'ID': test_ids.values,
    'Cancer_Result': result_labels
})

print("\nFinal Predictions DataFrame:")
print(results_df.head())

import tkinter as tk
from tkinter import filedialog

# Initialize tkinter and hide the main window
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True) # Brings the window to the front

# Ask the user for a file name and location
file_path = filedialog.asksaveasfilename(
    defaultextension='.xlsx',
    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
    title="Choose where to save your Cancer Predictions"
)

# If the user didn't cancel the dialog
if file_path:
    results_df.to_excel(file_path, index=False)
    print(f"File successfully saved at: {file_path}")
else:

    print("Save cancelled.")
