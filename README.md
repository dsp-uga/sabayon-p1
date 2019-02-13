# sabayon-p1

## Table of contents

1. [Purpose](#purpose)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Usage](#usage)
5. [License](#license)
6. [Contact info](#contact-info)

## Purpose

sabayon-p1 creates a classifier to classfiy malware for project 1, course CSCI8360 of UGA. 

The project addresses the malware detection problem in the context of text classification.

## Requiremnets

This project requires Python 3.x with `pyspark` library installed.

## Dataset

The dataset consists of a set of documents belong to 9 different malware families. The content of the documents is in hexadecimal.

Since the size of the dataset is large, the data is not included in the repository, however a script to download a sample of the data is provided in the repository. To download the data navigate to `scripts/` directory and run on of the follwoings in command line (note that [`google-cloud-sdk`](https://cloud.google.com/sdk/) is required for the script): 

`$ ./get_files.sh`

`$ python3 get_files.py`

## Usage

To use the program simply run the following command:

`$ python3 classify.py`

List of command line arguments to pass to the program are as follows:

	--asm_path: path to the asm training files.
	--bytes_path: path to the bytes training files.
	--train_files: path to the file containing the train files.
	--test_files: path to the file containing the test files.
	--train_labels: path to the file containing the train labels.
	--test_labels: path to the file containing the test labels.
	--outfile: path to the output file containing labels for final test set.
	--model_path: path to the folder for saving the final model.
	--n_parts: an integer specifying the number of partitions.
	--mem_lim: a string specifying the memory limit.
	--max_depth: maximum depth of the tree in Rnadom Forest Classifier.
	--classifier: classifier algorithm to be used for the classification task ({lr,nb,rf}).

The default values for above parameters are set such that it builds a Random Forest Classifier with `max_depth=7` and works with [public data](https://console.cloud.google.com/storage/browser/uga-dsp/project1) available on Google cloud 

The see the above list in command line execute the following command:

`$ python3 classify.py -h`

One typical usage is:

`$ python3 classify.py --bytes_path="../dataset/data/bytes/" --train_files="../dataset/files/X_small_train.txt" --test_files="../dataset/files/X_small_test.txt" --train_labels="../dataset/files/y_small_train.txt" --outfile="./outfile.csv" --mem_lim="4G" --asm_path=""`

Note that the program works with both *asm* and *bytes* data, so only one of the `asm_path` and `bytes_path` should be a valid path and the other one should be an empty string.

## License
The code in this repository is free software: you can redistribute it and/or modify it under the terms of the MIT lisense. 

## Contact info

For questions please email one of the authors: 

**saedr@uga.edu**

**marcdh@uga.edu**

**jayant.parashar@uga.edu**
