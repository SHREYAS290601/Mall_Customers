import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans

sb.set()
sb.set_style("darkgrid")
# This code is for viewing a lot of information on the console
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

# Main code
# Reading the csv file
df = pd.read_csv("C:\\Users\\Shreyas\\Mall_Customers.csv")
df.drop(["CustomerID"], axis=1, inplace=True)
# print(df)

# For viewing many graphs at the same time
fig, axs = plt.subplots(2, 3)

# Violinplot is the best graph type here as it requires age visualisation only
ageFrequency = sb.violinplot(y=df['Age'], color='red', ax=axs[0, 0])
anualIncome = sb.violinplot(y=df['Annual Income (k$)'], color='blue', ax=axs[0, 1])
spendingScores = sb.boxplot(y=df['Spending Score (1-100)'], color='green', ax=axs[0, 2])

# Male vs Female
genders = df['Gender'].value_counts()
MaleVsFemale = sb.barplot(x=genders.index, y=genders.values, ax=axs[1, 0],label='Gender Visualization')
axs[1,0].set_xlabel('Gender Visualization')
#

# Segregating the age according to different age groups
age0to20 = df.Age[(df.Age <= 20)]
age21to40 = df.Age[(df.Age <= 40) & (df.Age >= 21)]
age41to60 = df.Age[(df.Age <= 60) & (df.Age >= 41)]
age60andAbove = df.Age[df.Age >= 61]
x = ["0-20", "21-40", "41-60", "60 and above"]
y = [len(age0to20), len(age21to40), len(age41to60), len(age60andAbove)]
ageDistributionOfCustomers = sb.barplot(x=x, y=y, palette="tab10", ax=axs[1, 1])
axs[1,1].set_xlabel('Age distribution')


# Segregating the income according to different income groups
income0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 0) & (df["Annual Income (k$)"] <= 30)]
income30_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] > 30) & (df["Annual Income (k$)"] <= 60)]
income60_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] > 60) & (df["Annual Income (k$)"] <= 90)]
income90_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] > 90) & (df["Annual Income (k$)"] <= 120)]
income120andAbove = df["Annual Income (k$)"][(df["Annual Income (k$)"] > 120) & (df["Annual Income (k$)"] <= 160)]

xx = ["0_30(k$)", "30_60(k$)", "60_90(k$)", "90_120(k$)", "120+(k$)"]
yy = [len(income0_30), len(income30_60), len(income60_90), len(income90_120), len(income120andAbove)]

incomePlotBar = sb.barplot(x=xx, y=yy, palette="Set2", ax=axs[1, 2])
plt.xlabel("Income of customers")
plt.show()

# Segregating the scores according to specific groups
ss0_20 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] > 0) & (df["Spending Score (1-100)"] <= 20)]
ss20_40 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] > 20) & (df["Spending Score (1-100)"] <= 40)]
ss40_60 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] > 40) & (df["Spending Score (1-100)"] <= 60)]
ss60_80 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] > 60) & (df["Spending Score (1-100)"] <= 80)]
ss80_100 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] > 80) & (df["Spending Score (1-100)"] <= 100)]

ssx = ["0-20", "20-40", "40-60", "60-80", "80-100"]
ssy = [len(ss0_20.values), len(ss20_40.values), len(ss40_60.values), len(ss60_80.values), len(ss80_100.values)]
spendingScoresBar=sb.barplot(x=ssx,y=ssy,palette='Set2')
plt.title("Spending Scores of the customers")
plt.show()

# Elbow Methos for finding the optimal fit for the dataset which is given to us
distortions = []
K = range(1, 11)
df2 = df.drop(['Gender'], axis=1)
print(df2)
for k in K:
    kmeans = KMeans(n_clusters=k,
                    init='k-means++')  # selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
    kmeans.fit(df2)
    distortions.append(kmeans.inertia_)
plt.plot(K,distortions,'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.grid()
plt.show()

# According to the graph of elbow method it was visible that for k=5 the clustering is possible thus we will use the same method
kMeans = KMeans(n_clusters=5)
cluster = kMeans.fit_predict(df2)  # Here we are predicting to which center or label will the dataset be associated
df2['labels'] = cluster
# print(df2)

LABEL_COLOR_MAP = {0: 'red',
                   1: 'yellow',
                   2: 'blue',
                   3: 'green',
                   4: 'purple'
                   }
label_color = [LABEL_COLOR_MAP[l] for l in df2.labels]
plot=sb.scatterplot(x=df2['Annual Income (k$)'], y=df['Spending Score (1-100)'], c=label_color)#here for c I have tried another shortcut method as df2.labels.astype(np.float) however the color scheme doesnt look good and one
#of the colors is nearly invisible.
plt.xlabel("Annual Income")
plt.ylabel('Spending Score')


figure = plt.figure()
Graph3D = figure.add_subplot(111, projection = '3d')
age=df2.Age
income=df2['Annual Income (k$)']
spendingScores=df2['Spending Score (1-100)']
Graph3D.set_xlabel('Annual Income (k$)')
Graph3D.set_ylabel('Spending Score (1-100)')
Graph3D.set_zlabel('Age')
Graph3D.scatter(income,spendingScores,age,c=label_color)
plt.show()
