from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import RegexTokenizer, Tokenizer, HashingTF, IDF, StopWordsRemover, NGram, Word2Vec 
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from os import path

#Spark Defaults
sc = SparkContext()
spark = SparkSession(sc)

#Create training filenames
train_names = sc.textFile('../../dataset/files/X_small_train.txt').collect()
file_path = 'file:' + path.realpath('../../../data/asm') + '/' 
for i in range(len(train_names)):
	train_names[i] = file_path + train_names[i] + '.asm'

#Create Training ID-Label Dict
train_labels = sc.textFile('../../dataset/files/y_small_train.txt').collect()
train_id_label = {}
for i in range(len(train_names)):
	train_id_label[train_names[i]] = train_labels[i]
train_id_label = sc.broadcast(train_id_label)

#Create Training Dataframe
data = sc.wholeTextFiles(','.join(train_names))
train_data = data.map(lambda x: (x[0], x[1], int(train_id_label.value[x[0]])))
train_df = train_data.toDF(['id', 'text', 'label'])

#Create testing filenames
test_names = sc.textFile('../../dataset/files/X_small_test.txt').collect()
file_path = 'file:' + path.realpath('../../../data/asm') + '/'
for i in range(len(test_names)):
	test_names[i] = file_path + test_names[i] + '.asm'

#Create ID-Label Dict
test_labels = sc.textFile('../../dataset/files/y_small_test.txt').collect()
test_id_label = {}
for i in range(len(test_names)):
	test_id_label[test_names[i]] = test_labels[i]

test_id_label = sc.broadcast(test_id_label)

#Create Testing Dataframe
data = sc.wholeTextFiles(','.join(test_names))
test_data = data.map(lambda x: (x[0], x[1], int(test_id_label.value[x[0]])))
test_df = test_data.toDF(['id', 'text', 'label'])

#Training: Tokenize, W2V, Logistic Regression
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern='\w{8}|\s')
#tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol='words', outputCol='filtered', stopWords=['??', '.text', '.data'])
ngram = NGram(n=2, inputCol='filtered', outputCol='ngrams')
hashingTF = HashingTF(inputCol="ngrams", outputCol="features") #, numFeatures=256)
#idf = IDF(inputCol='freqs', outputCol='features')
#word2vec = Word2Vec(inputCol='ngrams', outputCol='features')
rf = RandomForestClassifier() #maxDepth=7

#ML Pipeline Model
pipeline = Pipeline(stages=[tokenizer, remover, ngram, hashingTF, rf])
model = pipeline.fit(train_df)
model.save('RF_Bigram_TF_5_ASM')
predictions = model.transform(test_df)

#Evaluate Model Accuracy
test_predictions = predictions.select('id', 'prediction').collect()
correct = 0
for i in range(len(test_predictions)):
	if test_predictions[i][1] == int(test_id_label.value[test_predictions[i][0]]):
		correct += 1

print('RF Model Accuracy ', correct * 1.0 / len(test_predictions))
