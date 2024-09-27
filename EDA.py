import pandas as pd
import numpy as np
#from sklearn import
import matplotlib
from matplotlib import pyplot as plt

Data = pd.read_csv("clinical_mastitis_cows.csv")
print(Data)

Data_Type = Data.dtypes
print(Data_Type)

Data_Describe = Data.describe()
print(Data_Describe)

Breed_unique = Data["Breed"].value_counts()
print(Breed_unique)

plt.box(Data)
plt.show()

Breed_Jersey_Only = Data.drop(Data[Data["Breed"] == "holstene"].index,axis=0)
print(Breed_Jersey_Only)

Months_After_Birth = Data["Months after giving birth"].value_counts()
print(Months_After_Birth)

Previous_Mastits_status = Data["Previous_Mastits_status"].value_counts()
print(Previous_Mastits_status)

Data = Data.drop(["Breed","House Number","Address"], axis=1)
print(Data)