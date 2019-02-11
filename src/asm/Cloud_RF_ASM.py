from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import RegexTokenizer, HashingTF, StopWordsRemover, NGram 
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

#Spark Defaults
sc = SparkContext()
spark = SparkSession(sc)

#Create training filenames
train_names = sc.textFile('gs://uga-dsp/project1/files/X_train.txt').collect()
file_path = 'gs://uga-dsp/project1/data/asm/' 
for i in range(len(train_names)):
	train_names[i] = file_path + train_names[i] + '.asm'

#Create ID-Label Dict
train_labels = sc.textFile('gs://uga-dsp/project1/files/y_train.txt').collect()
train_id_label = {}
for i in range(len(train_names)):
	train_id_label[train_names[i]] = train_labels[i]
train_id_label = sc.broadcast(train_id_label)

#Create Training Dataframe
data = sc.wholeTextFiles(','.join(train_names), 50)
train_data = data.map(lambda x: (x[0], x[1], int(train_id_label.value[x[0]])))
train_df = train_data.toDF(['id', 'text', 'label'])

#Create testing filenames
test_names = sc.textFile('gs://uga-dsp/project1/files/X_test.txt').collect()
file_path = 'gs://uga-dsp/project1/data/asm/'
for i in range(len(test_names)):
	test_names[i] = file_path + test_names[i] + '.asm'

#Create Training Dataframe
test_data = sc.wholeTextFiles(','.join(test_names), 50)
test_df = test_data.toDF(['id', 'text'])

#Training: Tokenize, W2V, Logistic Regression
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern='\w{8}|\s')
remover = StopWordsRemover(inputCol='words', outputCol='filtered', stopWords=['??', '.text', '.data'])
ngram = NGram(n=2, inputCol='filtered', outputCol='ngrams')
hashingTF = HashingTF(inputCol="ngrams", outputCol="features")
rf = RandomForestClassifier(maxDepth=7)

#ML Pipeline Model
pipeline = Pipeline(stages=[tokenizer, remover, ngram, hashingTF, rf])
model = pipeline.fit(train_df)
model.save('gs://malware-classifier-p1/RF_Bigram_TF_7_large_asm')
predictions = model.transform(test_df)

#Prediction Output
test_pred = predictions.select('id', 'prediction').collect()
test_names = open('/home/marcus/X_test.txt')
file_path = 'gs://uga-dsp/project1/data/asm/'
for i in range(len(test_names)):
	test_names[i] = file_path + test_names[i] + '.asm'

id_label = {}
for i in range(len(test_pred)):
	id_label[test_pred[i][0]] = test_pred[i][1]

pred_str = ''
for i in range(len(test_names)):
	pred_str += str(int(id_label[test_names[i]])) + '\n'

writer = open('/home/marcus/Pred_RF_ASM.txt', 'w')
writer.write(pred_str)
writer.close()
