import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.svm import SVR

test=np.load('date31.npy')
length=test.size;
feature=[[] for i in range(length)]
for i in range(14,31):
    filename='date'+str(i)+'.npy'
    for j in range(length):
        feature[j].append(np.load(filename)[j])
for i in range(length):
    feature[i]=np.array(feature[i])


svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.0001)
result=[test[0]]

for i in range(length-1):
    target=np.array(test[i])
    y_rbf = svr_rbf.fit(feature[i].reshape(1,feature[i].size),np.ravel(target))  #target.reshape(1,1)
    result.append(float(y_rbf.predict(feature[i+1].reshape(1,feature[i+1].size))))

time=[i for i in range(length)]



fig, ax = plt.subplots(1,1)
ax.scatter(time,test , color='darkorange', label='data')
#plt.plot(time,test[:-1] , color='darkorange', label='data')
ax.hold('on')
#plt.scatter(time, result, color='navy', label='RBF model')
ax.plot(time, result, color='navy', label='RBF model')
ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
plt.ylabel('Total_power')
plt.xlabel('prediction in seconds')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

#Error analysis for 0.1%
count=0
test=list(test)
#result=result[:-1]
if len(test) != len(result):
        raise ValueError
for i in range(len(result)):
    if abs(test[i]-result[i])<=100:
        count=count+1
print('the program correctly predict about '+str(count/len(test)*100)+' % of total test data')
print('The computation is performed under the following error rate: '+str(100/(max(test)-min(test))))

#Error analysis for 1%
count=0
test=list(test)
#result=result[:-1]
if len(test) != len(result):
        raise ValueError
for i in range(len(result)):
    if abs(test[i]-result[i])<=1000:
        count=count+1
print('the program correctly predict about '+str(count/len(test)*100)+' % of total test data')
print('The computation is performed under the following error rate: '+str(1000/(max(test)-min(test))))





