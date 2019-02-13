import argparse
import sys

from pyspark.ml import Pipeline

from helper import spark_session_setup
from helper import load_dataset, write_to_file
from helper import write_to_file
from helper import build_pipeline

def main():
	# reading the command line arguments
	parser = argparse.ArgumentParser(description='Read in file paths and other parameters.')
	parser.add_argument('--asm_path', help='path to the asm training files.', default="gs://uga-dsp/project1/data/asm/", type=str)
	parser.add_argument('--bytes_path', help='path to the bytes training files.', default="gs://uga-dsp/project1/data/bytes/", type=str)
	parser.add_argument('--train_files', help='path to the file containing the train files.', default="gs://uga-dsp/project1/files/X_train.txt", type=str)
	parser.add_argument('--test_files', help='path to the file containing the test files.', default="gs://uga-dsp/project1/files/X_test.txt", type=str)
	parser.add_argument('--train_labels', help='path to the file containing the train labels.', default="gs://uga-dsp/project1/files/y_train.txt", type=str)
	parser.add_argument('--test_labels', help='path to the file containing the test labels.', default="gs://uga-dsp/project1/files/y_train.txt", type=str)
	parser.add_argument('--outfile', help='path to the output file containing labels for final test set.', default="gs://p1-models/RF_Large_Predictions.csv", type=str)
	parser.add_argument('--model_path', help='path to the folder for saving the final model.', default="gs://models/", type=str)
	parser.add_argument('--n_parts', help='an integer specifying the number of partitions.', default=50, type=int)
	parser.add_argument('--mem_lim', help='a string specifying the memory limit.', default='10G', type=str)
	parser.add_argument('--max_depth', help='maximum depth of the tree in Rnadom Forest Classifier.', default=7, type=int)
	parser.add_argument('--classifier', choices=['lr','nb','rf'], help='classifier algorithm to be used for the classification task.', default='rf', type=str)
	args = parser.parse_args()

	# initializing the variables
	print("Initializing the variables....")
	asm_path = args.asm_path
	bytes_path = args.bytes_path
	train_files = args.train_files
	test_files = args.test_files
	train_labels = args.train_labels
	test_labels = args.test_labels
	outfile = args.outfile
	model_path = args.model_path
	n_parts = args.n_parts
	memory_limit = args.mem_lim
	max_depth = args.max_depth
	classifier = args.classifier

	sc = spark_session_setup(memory_limit=memory_limit)

	# loading the dataset
	print("loading the dataset...")
	train_df, test_df = load_dataset(sc, asm_path=asm_path, bytes_path=bytes_path,
									 X_train=train_files, y_train=train_labels,
									 X_test=test_files, y_test=test_labels, n_parts=n_parts)

	# building the model
	print("building the model...")
	stages = build_pipeline(classifier=classifier, max_depth=max_depth)
	pipeline = Pipeline(stages=stages)
	model = pipeline.fit(train_df)

	# saving the model and writing the predictions into the output file
	if model_path:
		model.save(model_path)
	print("generatign the predictions...")
	predictions = model.transform(test_df)
	write_to_file(predictions, outfile)

if __name__ == '__main__':
	sys.exit(main())
