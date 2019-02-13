import setuptools

with open("README.md", 'r') as fp:
	long_description = fp.read()

setuptools.setup(
	name = "sabayon-p1",
	version = "1.0.0",
	author="Saed Rezayi, Marcus Hill, Jayant Parashar",
	author_email="saedr@uga.edu, marcdh@uga.edu, jayant.parashar@uga.edu",
	license='MIT',
	description="A package for malware classification.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/dsp-uga/sabayon-p1",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
	],
	test_suite='nose.collector',
	tests_require=['nose'],
    install_requires=['pyspark'],
)
