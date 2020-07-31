# https://www.digitalocean.com/community/tutorials/how-to-work-with-language-data-in-python-3-using-the-natural-language-toolkit-nltk

import random
import re
import string

from nltk.corpus import twitter_samples
import nltk
from nltk.tag import pos_tag
from nltk.tag import pos_tag_sents
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize


# region  Utility Functions Ctrl + Alt + T
def download_nltk_resources():
    nltk.download('punkt')  # a pre-trained model that helps you tokenize words and sentences
    nltk.download('wordnet')  # lexical database for the English language that
    # helps the script determine the base word
    nltk.download('averaged_perceptron_tagger')  # to determine the context of a word in a sentence
    nltk.download('stopwords')


def remove_noise(tweet_tokens, stop_words = ()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def lemmatize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startsWith('NN'):
            pos = 'n'
        elif tag.startsWith('VM'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    temp = dict()
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)
    #     for token in tweet_tokens:
    #         temp.update(token, True)
    # yield temp
# endregion


def samain():
    # Test the environment
    # twitter_samples.fileids()
    # ['negative_tweets.json', 'positive_tweets.json', 'tweets.20150430-223406.json']
    # twitter_samples.strings('tweets.20150430-223406.json')
    # Testing is done, not the real thing
    # Step 1. Download corpora
    download_nltk_resources()
    # Step 2. Tokenizing the Data
    positive_tweets = twitter_samples.strings('positive_tweets.json')  # 5000 tweets with negative sentiments
    negative_tweets = twitter_samples.strings('negative_tweets.json')  # 5000 tweets with positive sentiments
    text = twitter_samples.strings('tweets.20150430-223406.json')  # 20000 tweets with no sentiments

    # now we need to tokenize our tweets
    # (break up sequences into individual words, phrases, symbols, etc.
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    print("\nPositive Tweet Tokens:\n", pos_tag(positive_tweet_tokens[0]))
    print("\nNegative Tweet Tokens:\n", pos_tag(negative_tweet_tokens[0]))
    # tweets_tokens is a list where each element in the list is a list of tokens
    # this is a list of lists
    # next we will tag each of our tokens
    tweets_tagged = pos_tag_sents(positive_tweet_tokens)
    # now each token is tagged as part of speech (POS) and this is how it looks:
    # [(u'#FollowFriday', 'JJ'), (u'@France_Inte', 'NNP'), (u'@PKuchly57', 'NNP'),
    # (u'@Milipol_Paris', 'NNP'), (u'for', 'IN'), (u'being', 'VBG'), (u'top', 'JJ'),
    # (u'engaged', 'VBN'), (u'members', 'NNS'), (u'in', 'IN'), (u'my', 'PRP$'),
    # (u'community', 'NN'), (u'this', 'DT'), (u'week', 'NN'), (u':)', 'NN')]
    # Each token/tag pair is saved as a tuple
    # In NLTK, the abbreviation for adjective is JJ.
    # singular nouns (NN)
    # plural nouns (NNS)
    # next let's count adjectives and singular nouns
    JJ_count = 0
    NN_count = 0
    for tweet in tweets_tagged:
        for pair in tweet:
            tag = pair[1]
            if tag == 'JJ':
                JJ_count += 1
            elif tag == 'NN':
                NN_count += 1

    print('Total number of adjectives = ', JJ_count)
    print('Total number of nouns = ', NN_count)

    # Step 3 Normalize the Data
    # Normalization in NLP is the process of converting a word to its canonical form

    # Step 4. Remove noise from the data
    stop_words = stopwords.words('english')
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    # Step 5 Determine Word Density
    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    all_neg_words = get_all_words(negative_cleaned_tokens_list)
    freq_dist_pos = FreqDist(all_pos_words)
    freq_dist_neg = FreqDist(all_neg_words)

    print("\nMost common positive: \n", freq_dist_pos.most_common(10))
    print('\n')
    print("Most commot negative:\n", freq_dist_neg.most_common(10))

    # Step 6 Preparing Data for the Model
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)  # Randomize records

    train_data = dataset[:7000]  # The first 7000 tweets is for training
    test_data = dataset[7000:]   # The remaining 3000 tweets is testing/evaluating the model

    # Step 7 Building and Testing The Model
    classifier = NaiveBayesClassifier.train(train_data)
    print("\nAccuracy:\n", classify.accuracy(classifier, test_data))
    # print("\nMost informative features:\n", classifier.show_most_informative_features(10))

    # Now evaluate the model on a new tweet:
    sample_text = "[SAMPLE TWEET]\n I ordered just once from TerribleCo, they screwed up, never used the app again."
    custom_tokens = remove_noise(word_tokenize(sample_text))

    print("\nClassify new text:\n" + sample_text + "\n", classifier.classify(dict([token, True] for token in custom_tokens)))

    sample_text = "[SAMPLE POST]\n Does anybody else feel that Representative Ted Yoho’s recent behavior is " \
                  "unbecoming of the profession? He was acting as a politician not a vet at the " \
                  "time that he verbally abused his female co- worker, but he is an active " \
                  "licensed equine vet from FL. If you agree or not with his politics or hers- " \
                  "it should not matter. The AVMA PAC gave him $37k for his campaign."
    custom_tokens = remove_noise(word_tokenize(sample_text))

    print("\nClassify new text:\n" + sample_text + "\n",
          classifier.classify(dict([token, True] for token in custom_tokens)))

    sample_text = "[SAMPLE OFFICIAL RESPONSE]\n" \
        " Thank you for contacting us on this matter, Dr. Sivula. We hear your concerns and take " \
        " this very seriously. Use of obscenities, gender-based insults and other offensive language " \
        " does not reflect the values of the AVMA. " \
        " This situation continues to evolve, and we are monitoring it as it develops. While Representative " \
        " Yoho’s reported comments to Representative Ocasio-Cortez were made outside the context of his " \
        " credentials as a veterinarian, the AVMA believes in treating every person with respect. " \
        " We expect our members, whatever their current role may be, to demonstrate professional " \
        " respect to others at all times. " \
        " We spoke with Rep. Yoho yesterday and have conveyed the depth of feeling on this issue " \
        " expressed by the AVMA members who have reached out to us. Our understanding is that he is " \
        " scheduled to speak to this issue tonight at 7 P.M. on CNN. " \
        "  We expect our members and leaders to exhibit professional behavior and we stand for the " \
        " right of all individuals to respectful treatment. " \
        " Thank you again for reaching out to us. We always appreciate hearing from our members. " \
        " Rena Carlson-Lammers, DVM " \
        " Chair, Board of Directors, American Veterinary Medical Association "

    custom_tokens = remove_noise(word_tokenize(sample_text))

    print("\nClassify new text:\n" + sample_text + "\n",
          classifier.classify(dict([token, True] for token in custom_tokens)))



if __name__ == '__main__':
    samain()
