
from pyspark import SparkContext
import sys
from os import path

sc = SparkContext()
data = sc.wholeTextFiles(sys.argv[1]) #'../../data/bytes'

#Training Set
fp = open('../dataset/files/X_small_train.txt')
train_names = fp.read().split()
file_path = 'file:' + path.realpath(sys.argv[1]) + '/'
for i in range(len(train_names)):
	train_names[i] = file_path + train_names[i] + '.bytes'
train_names = sc.broadcast(train_names)

#Training Labels
fp = open('../dataset/files/y_small_train.txt')
train_labels = sc.broadcast(fp.read().split())

train_data = data.filter(lambda x: x[0] in train_names.value)
