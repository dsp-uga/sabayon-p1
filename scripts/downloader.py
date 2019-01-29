#! /usr/bin/python3

import os

def X_small_train():
	reader = open('project1_files_X_small_train.txt')
	files = reader.read().split()

	for filename in files:
		os.system('wget https://storage.googleapis.com/uga-dsp/project1/data/bytes/' \
			+ filename + '.bytes' \
			+ ' -O ./bytes/X_small_train/' + filename + '.bytes')

def X_small_test():
	reader = open('project1_files_X_small_test.txt')
	files = reader.read().split()

	for filename in files:
		os.system('wget https://storage.googleapis.com/uga-dsp/project1/data/bytes/' \
			+ filename + '.bytes' \
			+ ' -O ./bytes/X_small_test/' + filename + '.bytes')

#Main
#X_small_train()
X_small_test()
