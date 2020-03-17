import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(m):
    return 1 / (1 + np.exp(-m))


dataSheet = pd.read_csv('bank.csv').values
n = len(dataSheet[:, 0])
age_data = dataSheet[:, 0].reshape(n, 1)
balance_data = dataSheet[:, 5].reshape(n, 1)
default_data = []
loan_data = []
job_data = []
marital_data = []
education_data = []
house_data = []
contact_data = []
duration_data = dataSheet[:, 9].reshape(n, 1)
p_out_cone_data = []
y = []

# job
for i in range(0, n):
    if dataSheet[i, 1] == 'admin.':
        job_data.append(1)
    elif dataSheet[i, 1] == 'unknown':
        job_data.append(2)
    elif dataSheet[i, 1] == 'unemployed':
        job_data.append(3)
    elif dataSheet[i, 1] == 'management':
        job_data.append(4)
    elif dataSheet[i, 1] == 'housemaid':
        job_data.append(5)
    elif dataSheet[i, 1] == 'entrepreneur':
        job_data.append(6)
    elif dataSheet[i, 1] == 'student':
        job_data.append(7)
    elif dataSheet[i, 1] == 'blue-collar':
        job_data.append(8)
    elif dataSheet[i, 1] == 'self-employed':
        job_data.append(9)
    elif dataSheet[i, 1] == 'retired':
        job_data.append(10)
    elif dataSheet[i, 1] == 'technician':
        job_data.append(11)
    elif dataSheet[i, 1] == 'services':
        job_data.append(12)
    else:
        job_data.append(0)

# marital
for i in range(0, n):
    if dataSheet[i, 2] == 'married':
        marital_data.append(1)
    elif dataSheet[i, 2] == 'divorced':
        marital_data.append(2)
    elif dataSheet[i, 2] == 'single':
        marital_data.append(3)
    else:
        marital_data.append(0)

# education
for i in range(0, n):
    if dataSheet[i, 3] == 'unknown':
        education_data.append(1)
    elif dataSheet[i, 3] == 'secondary':
        education_data.append(2)
    elif dataSheet[i, 3] == 'primary':
        education_data.append(3)
    elif dataSheet[i, 3] == 'tertiary':
        education_data.append(4)
    else:
        education_data.append(0)

# loan
for i in range(0, n):
    if dataSheet[i, 7] == 'no':
        loan_data.append(0)
    else:
        loan_data.append(1)

# default credit
for i in range(0, n):
    if dataSheet[i, 4] == 'no':
        default_data.append(0)
    else:
        default_data.append(1)

# house
for i in range(0, n):
    if dataSheet[i, 6] == 'no':
        house_data.append(0)
    else:
        house_data.append(1)

# contact
for i in range(0, n):
    if dataSheet[i, 8] == 'unknown':
        contact_data.append(1)
    elif dataSheet[i, 8] == 'telephone':
        contact_data.append(2)
    elif dataSheet[i, 8] == 'cellular':
        contact_data.append(3)

# p_outcome
for i in range(0, n):
    if dataSheet[i, 10] == 'unknown':
        p_out_cone_data.append(1)
    elif dataSheet[i, 10] == 'other':
        p_out_cone_data.append(2)
    elif dataSheet[i, 10] == 'failure':
        p_out_cone_data.append(3)
    elif dataSheet[i, 10] == 'success':
        p_out_cone_data.append(4)
    else:
        p_out_cone_data.append(0)

# result
for i in range(0, n):
    if dataSheet[i, 11] == 'no':
        y.append(0)
    else:
        y.append(1)

job_data = np.array(job_data).reshape(n, 1)
marital_data = np.array(marital_data).reshape(n, 1)
education_data = np.array(education_data).reshape(n, 1)
default_data = np.array(default_data).reshape(n, 1)
house_data = np.array(house_data).reshape(n, 1)
contact_data = np.array(contact_data).reshape(n, 1)
p_out_cone_data = np.array(p_out_cone_data).reshape(n, 1)
y = np.array(y).reshape(n, 1)

x = np.hstack((np.ones((n, 1)), age_data, job_data, marital_data, education_data, default_data, balance_data,
               house_data, contact_data, duration_data, p_out_cone_data))
x = np.array(x)
w = pd.read_csv('w_pre.csv', header=None).values
#print(w)
Iteration = 100
cost = np.zeros((1, n))
learning_rate = 0.001
cost = []

for i in range(0, Iteration):
    y_predict = sigmoid(np.dot(x, w).astype(object).astype(float))
    cost.append(np.multiply(y, np.log(y_predict)) + np.multiply((1 - y), np.log(1 - y_predict)))
    w = w - learning_rate * np.dot(x.T, y_predict - y)


x = [1.0, 30, 2, 1, 3, 1, 2323, 1, 3, 324, 2]
print(sigmoid(np.dot(x, w).astype(object).astype(float)))
print(np.dot(x, w))
#print(w)
#print(cost)
a = np.array(w).tolist()
np.savetxt("w_pre.csv", a)
