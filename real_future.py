# Multilayer Perceptron to Predict International Airline Passengers (t+1, given t, t-1, t-2)
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense

# convert an array of values into a dataset matrix
def create_dataset_for_exel(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a[0])
        b=dataset[i + look_back, 0]
        dataY.append(b)
    return dataX, dataY

def create_dataset(dataset, look_back=1):#used for creating dataset for ML
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        b=dataset[i + look_back, 0]
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)
	
	

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('GOOGL_stock_2017-05-07.csv', usecols=[1], engine='python', skipfooter=3)
#dataframe['Close']=dataframe['Close']/100
print(dataframe.head())
dataset = dataframe.values
dataset = dataset.astype('float32')
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#print(train)
print("THIS IS THE TEST DATASET",test)
# reshape dataset

look_back = 10
exel_trainX, exel_trainY = create_dataset_for_exel(train, look_back)
exel_testX, exel_testY = create_dataset_for_exel(test, look_back)
print(exel_testX)


#-----------
print(len(exel_trainY))
lst1 = exel_trainX
lst2 = exel_trainY
percentile_list = pd.DataFrame(
    {'TrainX': lst1,
     'trainY': lst2
    })
    
percentile_list.to_csv('train google.csv')
lst1 = exel_testX
lst2 = exel_testY
percentile_list = pd.DataFrame(
    {'TrainX': lst1,
     'trainY': lst2
    })
#print(percentile_list)
percentile_list.to_csv('test google.csv')

trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
#print(trainX)

print("THIS IS TESTx",testX)
print(len(testX))
print("\n")
testX=list(testX)

#testX.pop()
#print(testX)
i=len(testX)-1
list_needed=testX[i]
list_needed=list(list_needed)
print(list_needed)

x=len(list_needed)
print(x)
y=list_needed[len(list_needed)-1]
'''
while len(list_needed)>1:
    q=[y]*x
    for i in range(len(list_needed)-1):
        q[i]=list_needed[i+1]
    testX.append(list(q))   
    list_needed.pop(0)
testX.pop(0)
#print(testX)

'''

print(len(testX))
print(len(testY))
testY=list(testY)
'''
for i in range(look_back-2):
    testY.append(0)
'''
testY.append(0)   
print(len(testY))
testY=numpy.array(testY)
print(testY)
print(testX)
#-------Real Future
last_value=list(test[-2])
l=list(test)[-12:-2]
l2=[]
for i in l:
    l2.append(list(i)[0])
print("hiiiii",l2)
print("this is the last value",last_value)
list_needed.pop(0)
l2.append(last_value[0])
print(l2)
testX.append(l2)
testX=numpy.array(testX)


  
    
    

# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(12, input_dim=look_back, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=2, verbose=2)
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
#testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
print(numpy.array(list_needed))
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting

diff=[]
ratio=[]

p = model.predict(testX)
for u in range(len(testY)):
    pr = p[u][0]
    ratio.append((testY[u]/pr)-1)
    diff.append(abs(testY[u]- pr))
    #print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))
# plot baseline and predictions
'''
plt.plot(p,color='red', label='prediction')
plt.plot(testY,color='blue', label='testY')
plt.legend(loc='upper left')
plt.show()
'''
'''
plt.plot(dataset)
#plt.plot(testPredictPlot)
plt.show()
'''

