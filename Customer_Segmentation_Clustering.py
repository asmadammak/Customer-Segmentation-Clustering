import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics 
from mpl_toolkits.mplot3d import Axes3D  # Importing Axes3D from mpl_toolkits.mplot3d


# Load the CSV file into a DataFrame
df = pd.read_csv(r"C:\Users\ASMA DAMMAK\Desktop\Customer segmenation Project\marketing_campaign.csv", sep='\t')

# Check the data to ensure it's correctly loaded
print(df.head())

# Rename the columns
df.columns = ['ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome',
              'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
              'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
              'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3',
              'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain',
              'Z_CostContact', 'Z_Revenue', 'Response']

# Drop the 'ID' column
df.drop('ID', axis=1, inplace=True)

# Check for missing values and handle them if needed
df['Income'].replace('', np.nan, inplace=True)
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')

# Normalize 'Year_Birth', 'Income', and 'Recency'
scaler = MinMaxScaler()
df[['Year_Birth', 'Income', 'Recency']] = scaler.fit_transform(df[['Year_Birth', 'Income', 'Recency']])

# Map categorical variables to numerical values
education_mapping = {'Basic': 1, '2n Cycle': 2, 'Graduation': 3, 'Master': 4, 'PhD': 5}
marital_status_mapping = {'Single': 1, 'Together': 2, 'Married': 2, 'Divorced': 1, 'Widow': 1,
                          'Alone': 1, 'Absurd': 1, 'YOLO': 1}
df['Education'] = df['Education'].map(education_mapping)
df['Marital_Status'] = df['Marital_Status'].map(marital_status_mapping)

# Convert 'Dt_Customer' to datetime and calculate customer lifespan
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
today_date = datetime.now()
df['Customer_Lifespan'] = (today_date - df['Dt_Customer']).dt.days

# Normalize 'Customer_Lifespan'
df['Customer_Lifespan'] = scaler.fit_transform(df['Customer_Lifespan'].values.reshape(-1, 1))

# Normalize other numerical columns
columns_to_normalize = ['MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                        'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
                        'NumStorePurchases', 'NumWebVisitsMonth']
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

# Convert boolean columns to integers
boolean_columns = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Response']
df[boolean_columns] = df[boolean_columns].astype(int)

# Calculate 'Family_Size' and drop unnecessary columns
df['Family_Size'] = df['Kidhome'] + df['Teenhome'] + df['Marital_Status']
df.drop(['Dt_Customer', 'Kidhome', 'Teenhome', 'Marital_Status'], axis=1, inplace=True)

# Print the cleaned and transformed DataFrame
print(df.head())
print(df.info())

#Check for NaN Valus and drop them if existed 
nan_df = df.isnull().sum()

# Check for NaN values in specific columns
nan_columns = df.columns[df.isnull().any()]
nan_df_specific = df[nan_columns].isnull().sum()

# Print the DataFrame with NaN counts
print("NaN counts in the entire DataFrame:")
print(nan_df)

print("\nNaN counts in specific columns:")
print(nan_df_specific)
#Check total number of rows in the dataset
total_rows = df.shape[0]
print("Total number of rows in the dataset:", total_rows)

# Drop rows with NaN values in the 'Income' column
df.dropna(subset=['Income'], inplace=True)

# Check the updated DataFrame shape and NaN counts
print("Updated DataFrame shape after dropping NaN values:")
print(df.shape)

nan_count_income = df['Income'].isnull().sum()
print("NaN count in 'Income' column after dropping NaN values:", nan_count_income)


# features to be used for clustering
features_for_clustering = ['Year_Birth', 'Education', 'Income', 'Recency', 'MntWines', 'MntFruits', 
                           'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                           'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
                           'NumStorePurchases', 'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4',
                           'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Z_CostContact',
                           'Z_Revenue', 'Response', 'Customer_Lifespan', 'Family_Size']

X = df[features_for_clustering]

# Initialize empty lists to store inertia 
inertia_values = []

# Define a range of k values to test 
k_values = range(2, 6) 

# Iterate over each k value and fit the K-means model
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    inertia_values.append(kmeans.inertia_)

# Elbow Method graph
plt.figure(figsize=(10, 5))

# Plotting Inertia values
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k (Inertia)')

plt.tight_layout()
plt.show()

# Initialize K-means with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
# Fit K-means to the data
kmeans.fit(X)
# Get the cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to the DataFrame
df['Cluster'] = cluster_labels

# Visualize the clusters in a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for each cluster
for cluster_num in range(3):
    cluster_data = df[df['Cluster'] == cluster_num]
    ax.scatter(cluster_data['Income'], cluster_data['MntWines'], cluster_data['Family_Size'],
               label=f'Cluster {cluster_num}', alpha=0.7)

ax.set_xlabel('Income')
ax.set_ylabel('MntWines')
ax.set_zlabel('Family_Size')
ax.set_title('K-means Clustering (k=3)')
ax.legend()

plt.show()