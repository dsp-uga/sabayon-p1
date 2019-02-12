from os import path
import sys

import pyspark
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StopWordsRemover
from pyspark.ml.feature import  NGram, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier
from pyspark.ml import Pipeline


def spark_session_setup(memory_limit='10G'):
    """
    creates a spark context. 
    optional parameter is memory limit.
    >>> sc = spark_session_setup('12G')
    """

    # in order to be bale to change log level
    conf = pyspark.SparkConf()
    conf.set('spark.logConf', 'true')
    conf.set('spark.executor.memory', memory_limit)
    conf.set('spark.driver.memory', memory_limit)
    conf.set('spark.driver.maxResultSize', memory_limit)

    # create a spark session
    sc = pyspark.SparkContext(appName='word_count', conf=conf)

    # change log level to ERROR
    sc.setLogLevel("ERROR")
    spark = SparkSession(sc)
    return sc

def load_dataset(n_parts=50, asm_path='', bytes_path='', X_train='', y_train='', X_test='', y_test=''):
	"""
	a function to load either bytes or asm dataset.
	"""
	if asm_path and not bytes_path:
		file_path = asm_path
		extention = '.asm'
	elif bytes_path and not asm_path:
		file_path = bytes_path
		extention = '.bytes'
	else:
		raise ValueError("only asm_path or bytes_path should be passed.")

	train_names = sc.textFile(X_train).collect()
	for i in range(len(train_names)):
		train_names[i] = file_path + train_names[i] + extention

	#Create ID-Label Dict
	train_labels = sc.textFile(y_train).collect()
	train_id_label = {}
	for i in range(len(train_names)):
		train_id_label[train_names[i]] = train_labels[i]
	train_id_label = sc.broadcast(train_id_label)

	#Create Training Dataframe
	data = sc.wholeTextFiles(','.join(train_names), n_parts)
	train_data = data.map(lambda x: (x[0], x[1], int(train_id_label.value[x[0]])))
	train_df = train_data.toDF(['id', 'text', 'label'])

	#Create testing filenames
	test_names = sc.textFile(X_test).collect()
	for i in range(len(test_names)):
		test_names[i] = file_path + test_names[i] + extention

	#Create Training Dataframe
	test_data = sc.wholeTextFiles(','.join(test_names), 50)
	test_df = test_data.toDF(['id', 'text'])

	#Create test labels
	test_labels = sc.textFile(y_test).collect()

	return train_df, test_df, test_labels

