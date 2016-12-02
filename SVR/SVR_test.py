
# coding: utf-8

# In[17]:

get_ipython().run_cell_magic('writefile', 'dataBase.py', 'from pymongo import MongoClient,ASCENDING\nfrom time_trans import *\nimport datetime\nclass dataBase(object):\n    def __init__(self):\n        self.hostname="localhost"\n        self.port=27017\n        self.db=""\n        self.col=""\n        \n    def welcome(self):\n        print("###############")\n        print("speed-up-input!")\n        print("###############")\n        print("Please make sure that a MongoDB instance is running on a host!!!!!!!!\\n") \n\n    def preDB(self):\n        hostname=input("please input your hostname,d for default")\n        if hostname[0]==\'d\':\n            self.hostname="localhost"\n        port=input("please input port number,d for default\\n")\n        try:\n            port=int(port)\n        except Exception as err:\n            self.port=27017\n        self.db=input("please input your name of database: ")\n        self.col=input("please input your name of collections: ")\n        return self                         \n\n    def openDB(self,dataext):\n        client = MongoClient(self.hostname,self.port)\n        dbs_name=self.db\n        db=client[dbs_name]\n        collections_name=self.col\n        col=db[collections_name]\n        col.create_index([("timeStamp", ASCENDING)])\n        startDate=datetime_timestamp(input("please input startDate(YYYY-MM-DD HH:MM:SS): "))\n        endDate=datetime_timestamp(input("please input endDate(YYYY-MM-DD HH:MM:SS): "))\n        cursor=col.find({\'timeStamp\':{\'$gte\': startDate,\'$lte\': endDate}})   \n        result=dataext.result(cursor)\n        return result\n        \n   \n    def __str__(self):\n        return self.hostname+"  "+str(self.port)')


# In[26]:

get_ipython().run_cell_magic('writefile', 'dataExtrc.py', 'import pandas as pd\nimport pickle\nimport math\nfrom time_trans import *\nclass dataExtrc(object):\n    def __init__(self):\n        self.real_power_matrix=[]\n        self.reactive_power_matrix=[]\n        self.total_power=[]\n        self.time_matrix=[]\n\n    def result(self,cursor):\n        n=0\n        for entry in cursor:\n            data = [pickle.loads(entry[\'rawData\'])][0]\n            self.real_power_matrix.append(sum(data[2]))\n            self.reactive_power_matrix.append(sum(data[3]))\n            self.total_power.append(math.sqrt(sum(data[2])**2+sum(data[3])**2))\n            #self.time_matrix.append(timestamp_datetime(data[-1]))\n            self.time_matrix.append(data[-1])\n            n+=1\n            if n>50000:\n                break\n\n        dict={"real_power":self.real_power_matrix,"reactive_power":self.reactive_power_matrix,\'total_power\':self.total_power}\n        result=pd.DataFrame(dict,index=self.time_matrix)\n        return result\n\n    def linear_approximation(self,startDate,control_const=100):\n        estimate_const=0.3\n        #60047 for a day/1786652 for whole database\n        max_Reach=control_const**2\n        default_Date="2016-01-13 07:00:00"\n        default_Date=datetime_timestamp(default_Date)\n        skip_entry=int((startDate-default_Date)*estimate_const)\n        if skip_entry>max_Reach:\n            skip_entry=max_Reach\n        if skip_entry<0:\n            skip_entry=0\n        return  skip_entry')


# In[22]:

get_ipython().run_cell_magic('writefile', 'time_trans.py', "import time\ndef timestamp_datetime(value):\n    format = '%Y-%m-%d %H:%M:%S'\n    value = time.localtime(value)\n    dt = time.strftime(format, value)\n    return dt\ndef datetime_timestamp(dt):\n    s = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))\n    return int(s)")


# In[1]:

from dataBase import *
from dataExtrc import *
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
user=dataBase()
user.welcome()
user.preDB()
dataext=dataExtrc()
result=user.openDB(dataext)

#plt.plot(result['total_power'])
#plt.show()
y_list=list(result['total_power'])
time_std=list(result.index)

x=np.array(time_std).T
y=np.array(result['total_power']).T

mean_x=np.mean(x)
std_x=np.std(x)

mean_y=np.mean(y)
std_y=np.std(y)

for i in range(len(list(result.index))):
    time_std[i]=(time_std[i]-mean_x)/std_x

for i in range(len(y)):
    y[i]=(y[i]-mean_y)/std_y

x=np.array(time_std).T.reshape(-1, 1)
length=len(y)

plt.plot(x,y,'r+')
plt.show()


svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.0001)
y_rbf = svr_rbf.fit(x, y).predict(x)

lw=2
plt.scatter(x, y, color='darkorange', label='data')
plt.plot(x, y_rbf, color='navy', lw=lw, label='RBF model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

res=y-y_rbf
counter=0
length=len(x)
print(length)
range_s=(max(y)-min(y))/100
for i in range(length):
    if abs(res[i])<range_s:
        counter=counter+1
print('if we allowed a error rate of 1%')
print('the successful rate is: '+str(counter*100.0/len(time_std))+"%")


# In[1]:

from dataBase import *
from dataExtrc import *
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np
user=dataBase()
user.welcome()
user.preDB()
dataext=dataExtrc()
result=user.openDB(dataext)

#plt.plot(result['total_power'])
#plt.show()
y_list=list(result['total_power'])
time_std=list(result.index)

x=np.array(time_std).T
y=np.array(result['total_power']).T

mean_x=np.mean(x)
std_x=np.std(x)

mean_y=np.mean(y)
std_y=np.std(y)

for i in range(len(list(result.index))):
    time_std[i]=(time_std[i]-mean_x)/std_x

for i in range(len(y)):
    y[i]=(y[i]-mean_y)/std_y

x=np.array(time_std).T.reshape(-1, 1)
length=len(y)

plt.plot(x,y,'r+')
plt.show()


svr_poly = SVR(kernel='poly', C=1e3, degree=3)
y_poly = svr_poly.fit(x, y).predict(x)

lw=2
plt.scatter(x, y, color='darkorange', label='data')
plt.plot(x, y_poly, color='navy', lw=lw, label='Poly model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

res=y-y_poly
counter=0
length=len(x)
#print(length)
range_s=(max(y)-min(y))/100
for i in range(length):
    if abs(res[i])<range_s:
        counter=counter+1
print('if we allowed a error rate of 1%')
print('the successful rate is: '+str(counter*100.0/len(time_std))+"%")

