
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
import sys
from os import path

def append_train_label(document):
	for i in range(len(train_names.value)):
		if document[0] == train_names.value[i]:
			return document[0], document[1], int(train_labels.value[i])

def match_test_label(document):
	for i in range(len(test_names.value)):
		if document[0] == test_names.value[i]:
			return int(test_labels.value[i])

#Spark Defaults
sc = SparkContext()
spark = SparkSession(sc)

#Training Set
data = sc.wholeTextFiles('../../data/bytes') #sys.argv[1]
fp = open('../dataset/files/X_small_train.txt')
train_names = fp.read().split()
file_path = 'file:' + path.realpath('../../data/bytes') + '/' #sys.argv[1]
for i in range(len(train_names)):
	train_names[i] = file_path + train_names[i] + '.bytes'
train_names = sc.broadcast(train_names)

#Training Labels
fp = open('../dataset/files/y_small_train.txt')
train_labels = sc.broadcast(fp.read().split())

#Convert Training Data into a Data Frame
train_data = data.filter(lambda x: x[0] in train_names.value)
train_data = train_data.map(append_train_label)
train_df = train_data.toDF(['id', 'text', 'label'])

#Testing Set
fp = open('../dataset/files/X_small_test.txt')
test_names = fp.read().split()
file_path = 'file:' + path.realpath('../../data/bytes') + '/' #sys.argv[1]
for i in range(len(test_names)):
	test_names[i] = file_path + test_names[i] + '.bytes'
test_names = sc.broadcast(test_names)

#Testing Labels
fp = open('../dataset/files/y_small_test.txt')
test_labels = sc.broadcast(fp.read().split())

#Convert Testing Data into a Data Frame
test_data = data.filter(lambda x: x[0] in test_names.value)
#test_data = test_data.map(append_test_label)
test_df = test_data.toDF(['id', 'text'])

#Training: Tokenize, Frequency, TF-IDF
tokenizer = Tokenizer(inputCol="text", outputCol="words")
#training_words = tokenizer.transform(train_df)
hashingTF = HashingTF(inputCol="words", outputCol="freqs", numFeatures=256)
#training_freq = hashingTF.transform(training_words)
idf = IDF(inputCol='freqs', outputCol='features')
#idf_model = idf.fit(training_freq)
#training_tfidf = idf_model.transform(training_freq)
nb = NaiveBayes(smoothing=1.0, modelType='multinomial')

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb])
model = pipeline.fit(train_df)
model.transform(test_df).show()
