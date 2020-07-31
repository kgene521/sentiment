# Sentiment Analysis

# Check if nltk is installed
$ python -c "import nltk"
# If no errors then yes, if errors install nltk:
$ pip install nltk
$ python -c "import nltk"
# download twitter_samples (or other text corpus):
$ python -m nltk.downloader twitter_samples

/*
D:\>python -m nltk.downloader twitter_samples
C:\Programs\Python\lib\runpy.py:125: RuntimeWarning: 'nltk.downloader' found in sys.modules after import of package 'nltk', but prior to execution of 'nltk.downloader'; this may result in unpredictable behaviour
  warn(RuntimeWarning(msg))
[nltk_data] Downloading package twitter_samples to
[nltk_data]     C:\Users\kgene\AppData\Roaming\nltk_data...
[nltk_data]   Unzipping corpora\twitter_samples.zip.
*/

# Next, download the part-of-speech (POS) tagger. 
# POS tagging is the process of labelling a word in a 
# text as corresponding to a particular POS tag: 
# nouns, verbs, adjectives, adverbs, etc. 

$ python -m nltk.downloader averaged_perceptron_tagger
# [nltk_data] Downloading package averaged_perceptron_tagger to
# [nltk_data]     C:\Users\kgene\AppData\Roaming\nltk_data...
# [nltk_data]   Package averaged_perceptron_tagger is already up-to-
# [nltk_data]       date!





