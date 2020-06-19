import pandas as pd

file = pd.read_csv('diabetes.csv')

file[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = file[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

file['Glucose'] = file['Glucose'].fillna(file['Glucose'].mean())
file['BloodPressure'] = file['BloodPressure'].fillna(file['BloodPressure'].mean())
file['SkinThickness'] = file['SkinThickness'].fillna(file['SkinThickness'].mean())
file['Insulin'] = file['Insulin'].fillna(file['Insulin'].mean())
file['BMI'] = file['BMI'].fillna(file['BMI'].mean())

x = file.iloc[:,0:8].values
y = file.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)