# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("/mnt/data/file-P4KpMZ5ib17R2zNsUjf4Qn")

# Basic Information
print("Shape of dataset:", df.shape)
print("\nColumn types and missing values:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe(include='all'))

# Handling missing values
print("\nMissing Values Before:")
print(df.isnull().sum())

# Fill missing Age with median, Embarked with mode, drop Cabin (too many missing)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

print("\nMissing Values After:")
print(df.isnull().sum())

# Visualizations
sns.set(style="darkgrid")

# Survival count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Gender vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# Class vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age Distribution
plt.figure(figsize=(8,4))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
