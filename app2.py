import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('Shop_rec_final.csv')
df.drop('Competitors',inplace=True,axis=1)
cols=['Schools','Colleges','Hospitals','Gyms','Boys_Hostels','Girls_Hostels','Turfs','Parking_Spaces']
for col in cols:
    df[col].fillna(df[col].mode()[0],inplace=True)
cols=['Rent_Cost','Daily_Footfall','Avg_Income','Shop_Area_SqFt','Dist_To_Transit']
for i in cols:
    df[i].fillna(df[i].mean(),inplace=True)
df['Area_Type'].fillna(df['Area_Type'].mode()[0],inplace=True)
df['Dominant_Age_Group'].fillna(df['Dominant_Age_Group'].mode()[0],inplace=True)
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
df['Area_Type']=lab.fit_transform(df['Area_Type'])
df['Dominant_Age_Group']=lab.fit_transform(df['Dominant_Age_Group'])
x=df.drop(['Best_Shop_Type'],axis=1)
y=df['Best_Shop_Type']
out_iqr = {}
for i in x:
    Q1=df[i].quantile(0.25)
    Q3=df[i].quantile(0.75)
    IQR=Q3-Q1
    Lower=Q1 - 1.5 * IQR
    Upper = Q3 +1.5 * IQR
    out=df[(df[i]<Lower) | (df[i]>Upper)]
    out_iqr[i]=out.shape[0]
out_iqr
for i in x:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    Lower = Q1 - 1.5 * IQR
    Upper = Q3 + 1.5 * IQR
    df=df[(df[i]>Lower) & (df[i]<Upper)]
Q1=df['Rent_Cost'].quantile(0.25)
Q3=df['Rent_Cost'].quantile(0.75)
IQR=Q3-Q1
Lower = Q1 - 1.5 * IQR
Upper = Q3 + 1.5 * IQR
df=df[(df['Rent_Cost']>Lower) & (df['Rent_Cost']<Upper)]
sns.boxplot(df['Rent_Cost'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test=scalar.transform(x_test)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100,random_state=2)
rfc.fit(x_train,y_train)
y_pred4=rfc.predict(x_test)
# Input: A single row of data (same format as x_train)
new_data = [[2, 1, 7, 1, 1, 1, 0, 2,
    15000, 680, 35000, 180, 120,
    0,1]]
probs = rfc.predict_proba(new_data)[0]

top3_idx = np.argsort(probs)[-3:][::-1]
classes = rfc.classes_
top3_classes = classes[top3_idx]
top3_probs = probs[top3_idx]

print("\nTop 3 Recommended Shops:\n")
for shop, p in zip(top3_classes, top3_probs):
    print(f"{shop}: {p*100:.2f}% success rate")