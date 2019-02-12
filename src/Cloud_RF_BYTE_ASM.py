from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import RegexTokenizer, HashingTF, StopWordsRemover, NGram 
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
import pyspark as ps
from pyspark import SparkContext
from pyspark import SQLContext
from pyspark import SparkConf
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import NaiveBayes
import sys
import requests
import re
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from pyspark.sql import functions
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import VectorAssembler



#Spark Defaults
sc = SparkContext()
spark = SparkSession(sc)

small_X_train='gs://uga-dsp/project1/files/X_small_train.txt'
small_Y_train='gs://uga-dsp/project1/files/y_small_train.txt'
small_X_test='gs://uga-dsp/project1/files/X_small_test.txt'
small_Y_test='https://storage.googleapis.com/uga-dsp/project1/files/y_small_test.txt'

X_train='gs://uga-dsp/project1/files/X_train.txt'
Y_train='gs://uga-dsp/project1/files/y_train.txt'
X_test='gs://uga-dsp/project1/files/X_test.txt'

#Create training filenames
train_names = sc.textFile(small_X_train).collect()
file_path_asm = 'gs://uga-dsp/project1/data/asm/' 
file_path_byte = 'gs://uga-dsp/project1/data/bytes/' 

for i in range(len(train_names)):
	train_names[i] = file_path_asm + train_names[i] + '.asm'

#Create ID-Label Dict
train_labels = sc.textFile(small_Y_train).collect()
train_id_label = {}

for i in range(len(train_names)):
	train_id_label[train_names[i]] = train_labels[i]

train_id_label = sc.broadcast(train_id_label)


def addByte_data_column(x): 
	path = file_path_byte+x[0]+'.bytes' 
	text1 = requests.get(path).text
	return(x[0], x[1],text1, int(train_id_label.value[x[0]])



#Create Training Dataframe
data = sc.wholeTextFiles(','.join(train_names), 90)
train_data = data.map(lambda x: addByte_data_column(x))
train_df = train_data.toDF(['id', 'text', 'text_bytes' 'label'])

#Create testing filenames
test_names = sc.textFile(small_X_test).collect()
file_path_asm = 'gs://uga-dsp/project1/data/asm/'
for i in range(len(test_names)):
	test_names[i] = file_path_asm + test_names[i] + '.asm'

#Create Testing Dataframe
test_data = sc.wholeTextFiles(','.join(test_names), 90)
test_df_asm = test_data.toDF(['id', 'text'])

#Training: Tokenize, W2V, random forest classifier 

tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern='\w{8}|\s')
tokenizer2 = RegexTokenizer(inputCol="text_bytes", outputCol="words_bytes", pattern='\w{8}|\s')

remover = StopWordsRemover(inputCol='words', outputCol='filtered', stopWords=['.text', '.data'])

cv = CountVectorizer(inputCol="filtered", outputCol="features_asm", minDF=2.0)

cv2 = CountVectorizer(inputCol="words_bytes", outputCol="features_bytes", minDF=2.0)

assembler = VectorAssembler(
    inputCols=["features_asm", "features_bytes"],
    outputCol="features")

#ngram = NGram(n=2, inputCol='filtered', outputCol='ngrams')
#hashingTF = HashingTF(inputCol="ngrams", outputCol="features")
rf = RandomForestClassifier(maxDepth=7)

#ML Pipeline Model
pipeline = Pipeline(stages=[tokenizer, tokenizer2,remover, cv,cv2, assembler, rf])
model = pipeline.fit(train_df)
#model.save('gs://malware-classifier-p1/RF_Bigram_TF_7_large_asm')
predictions = model.transform(test_df)

'''
#Prediction Output
test_pred = predictions.select('id', 'prediction').collect()
test_names = open('/home/marcus/X_test.txt')
file_path_asm = 'gs://uga-dsp/project1/data/asm/'
for i in range(len(test_names)):
	test_names[i] = file_path_asm + test_names[i] + '.asm'

id_label = {}
for i in range(len(test_pred)):
	id_label[test_pred[i][0]] = test_pred[i][1]

pred_str = ''
for i in range(len(test_names)):
	pred_str += str(int(id_label[test_names[i]])) + '\n'
	

#writer = open('/home/marcus/Pred_RF_ASM.txt', 'w')
#writer.write(pred_str)
#writer.close()
'''
#Evaluate Model Accuracy
test_predictions = predictions.select('prediction').collect()
correct = 0
for i in range(len(test_predictions)):
	if test_predictions[i][0] == matched_test_labels[i]:
		correct += 1
print('RF Model Accuracy ', (correct / len(test_predictions)))

