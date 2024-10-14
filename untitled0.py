import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1
data = pd.read_csv('Project_1_Data.csv')
df = pd.DataFrame(data)

# Step 2
X = df["X"]
Y = df["Y"]
Z = df["Z"]
Step = df["Step"]
summary_stats = data.describe()
print(summary_stats)

plt.plot(Step, X, label='X')
plt.plot(Step, Y, label='Y')
plt.plot(Step, Z, label='Z')
plt.title('Coordinates vs. Steps')
plt.ylabel('Coordinates')
plt.xlabel('Steps')
plt.legend()

plt.show()
plt.scatter(Step, X, label='X')
plt.scatter(Step, Y, label='Y')
plt.scatter(Step, Z, label='Z')
plt.title('Coordinates vs. Steps')
plt.ylabel('Coordinates')
plt.xlabel('Steps')
plt.legend()
plt.show()

sns.displot(X)
plt.show()
sns.displot(Y)
plt.show()
sns.displot(Z)
plt.show()

# Step 3
pearsoncorr = df.corr(method='pearson')
sns.heatmap(pearsoncorr, vmin=-1, vmax=1, annot=True)