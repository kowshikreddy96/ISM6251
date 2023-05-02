#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle
Lawnmover_model = pickle.load(open("C:/Users/dkrre/OneDrive/Desktop/ll/pickle.pkl", "rb"))

print("\n*******************")
print("* Prediction model for lawnmover ownership")
print("*******************\n")
Income = float(input("Enter the Income: "))
Lot_Size= float(input("Enter the Lotsize: "))
df = pd.DataFrame({'Income': [Income], 'Lot_Size': [Lot_Size]})
result = Lawnmover_model.predict(df)
probability = Lawnmover_model.predict_proba(df)
Ownership = ('Owner', 'Nonowner')
print(f"\n Prediction Model for lawnmover ownership indicates that the probability of Ownership is {probability[0][1]:.4f}, hence it is indicated as: {Ownership[result[0]]}.\n")


# In[ ]:




