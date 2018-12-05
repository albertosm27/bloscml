# Machine Learning for Blosc #

This is my [Final Degree Project](http://repositori.uji.es/xmlui/handle/10234/174211) whose objective is to explore the application of classification algorithms for the Blosc meta compressor.

### What is BloscML? ###

BloscML is a research project for the application of classification algorithms to automatically tune [Blosc](http://blosc.org/pages/blosc-in-depth/) compressor options.  
It consists of a set of scripts and jupyter notebooks which show the analysis of the compressible features of the data to apply supervised classification algorithms
for tuning all the configuration options of the Blosc compressor (codec, compression level, block size, pre-compression filters).


### How do I get set up? ###

Mainly you need to install Python with the packages python-blosc and scikit-learn.
The datasets used for the analysis unfortunately are too big to make them available here but all the results obtained from them are inside notebooks/data.
The interesting work for the "quick" readers is inside notebooks/deliver, unfortunately the work done is in Spanish (possibly translatable on request) 

### Who do I talk to? ###

Author: Alberto Sabater Morales alberto3pt@gmail.com  
Blosc organization: https://github.com/Blosc
