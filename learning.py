import sys
import pickle
import pandas as pd
import numpy as np
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
from nltk.tag import pos_tag, pos_tag_sents
from nltk import RegexpParser
from nltk import classify
from nltk import NaiveBayesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from pprint import pprint

# import nltk # used for downloading


if __name__ == "__main__" :
    
    #   Modules downloading.
    # download_dir = "/Users/yboadh/goinfre/nltk_data"
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('stopwords')
    # nltk.download('state_union', download_dir=download_dir)
    # nltk.download()

    ##************************************************ PART ONE ************************************************

    # text = "hello all this is my first nltk programe, i ll finish it, i think. i am programming using python. \
        # and pythoning should be done pythonly"

    # sent_tokenize : tokenize the input by sentences.
    # print (f"sentence toknizer : %s" % sent_tokenize(text))



    # word_tokenize : tokenize the input by words it take ',', '.', '?' ... as words too ! 

    # words = list(word_tokenize(text))
    # print(f"word tokenizer %s" % word_tokenize(text))




    # list of availablee stopword : u can specifie the language u want : Arabic, English, French ...

    # print (f"Stop word %s" % stopwords.words("english"))
    # stop_words = list(stopwords.words("english"))
    
    

    # Removing stop words from our word_tokenized text, and append them to new list.

    # clean_text = []
    # for word in words :
    #     if (word not in stop_words) :
    #         clean_text.append(word)
    #print(clean_text)

    
    """
        Steam* the clean word.
        Steam : it's like getting out the sufix of our words.
                for example (riding, sleeping, passage ...) => (ride, sleep, pass)
        
        the using of steaming may deppend on our use case 
        sometimes we might need it, other times no.
    """
    # steemed_words = []
    # ps = PorterStemmer()
    # for word in clean_text :
    #     steemed_words.append(ps.stem(word))
    # print(steemed_words)
    
    ##************************************************ PART TWO ************************************************

    # text = state_union.raw("2006-GWBush.txt")    
    # text_fd = open("text.txt", "r")
    # text = text_fd.read()

    """
        train the un-supervised PunkTokenizer Model.
        the model is pre-trained for english use (only) I guess.
    """

    # pk_sentence_tokenizer = PunktSentenceTokenizer(training_text)
    # pk_sentence_tokenizer = PunktSentenceTokenizer()
    # sent_tokenized_text = pk_sentence_tokenizer.tokenize(text)
    
    # stop_words = stopwords.words("english")
    # clean_tokens = []
    # for word in sent_tokenized_text :
    #     if (word not in stop_words) :
    #         clean_tokens.append(word)

    # tagged_words = list()
    # chunks = list()
    # try : 
    #     for sentence in clean_tokens:

    #         # word_tokenize : like we said above, split i sentences to words including (',', '.', ...)
    #         words = word_tokenize(sentence)
    #         """
    #             Pos_tag : is when we use the PunkSentenceTokenizer() to feed our model the data [tokenized word].
    #                         we will get back the same input but as tuples of ("word", "tag"*).
    #             * tags  : A POS tag (Part of speech) is a special label assigned to each token(word) in a sentence.
    #                       they are used to indicate the part of speech and often also grammatical categories
    #                       such as tense, number, (plural/singular)...
    #             list of tags :
    #                     CC coordinating conjunction
    #                     CD cardinal digit
    #                     DT determiner "a, the, every"
    #                     EX existential there (like: "there is" ... think of it like "there exists")
    #                     FW foreign word
    #                     IN preposition/subordinating conjunction "on, in, beside, about, above, onto ...."
    #                     JJ adjective 'big'
    #                     JJR adjective, comparative 'bigger'
    #                     JJS adjective, superlative 'biggest'
    #                     LS list marker 1)
    #                     MD modal could, will
    #                     NN noun, singular 'desk'
    #                     NNS noun plural 'desks'
    #                     NNP proper noun, singular 'Harrison'
    #                     NNPS proper noun, plural 'Americans'
    #                     PDT predeterminer 'all the kids'
    #                     POS possessive ending parent's
    #                     PRP personal pronoun I, he, she
    #                     PRP$ possessive pronoun my, his, hers
    #                     RB adverb very, silently,
    #                     RBR adverb, comparative better
    #                     RBS adverb, superlative best
    #                     RP particle give up
    #                     TO to go 'to' the store.
    #                     UH interjection errrrrrrrm
    #                     VB verb, base form take
    #                     VBD verb, past tense took
    #                     VBG verb, gerund/present participle taking
    #                     VBN verb, past participle taken
    #                     VBP verb, sing. present, non-3d take
    #                     VBZ verb, 3rd person sing. present takes
    #                     WDT wh-determiner which
    #                     WP wh-pronoun who, what
    #                     WP$ possessive wh-pronoun whose
    #                     WRB wh-abverb where, when
    #         """
    #         tagged_words = pos_tag(words)

    #         """
    #             We want to chunk everythin, except of (or chinking) VB* (VB, VBD, VBG, VBN, VBP. VBZ),
    #             DT, TO.

    #         """
            
    #         chunk_struct =  r"""
    #                             Chunk : {<.*>+}
    #                                     }<DT|TO|IN>+{
    #                          """
            
    #         """
    #             uncomment to view the chunks that we made with just this simple regex.
    #             we did catche some pretty good, well structured chunks.
    #         """

    #         chunk_parser = RegexpParser(chunk_struct)
    #         local_chunk = chunk_parser.parse(tagged_words)
    #         chunks.append(local_chunk)
    #         print(local_chunk)
    #         local_chunk.draw()
    # except Exception as err :
    #     print(err)


        ##************************************************ PART TWO ************************************************

    """
        In this part we will review the wrapped NLTK(Scikit-Learn)
        But first we will try the naive bayes that  work {posterior = ((prior occurence) * liklihood) / evidence}
            This model even if it's simple and easly computed. he proved good outputs in real word situation's due to
        his scalability.
    """
    columns_names = ['sentiment', 'twt_id', 'date', 'flag', 'user', 'body']
    ds_tweets = pd.read_csv("/home/xox/my_work/data_sets/tweets.csv", encoding="ISO-8859-1", names=columns_names)
    # ds_tweets.drop(ds_tweets[ds_tweets.sentiment == 2].index, inplace=True)
    # stop_set = stopwords.words('english')
    # vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents="ascii", stop_words=stop_set)

    y = ds_tweets.sentiment
    # X = vectorizer.fit_transform(ds_tweets.body)
    # X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.25)

    # Mnom_classifier = MultinomialNB()
    # Mnom_classifier.fit(X_train, y_train)
    # print(roc_auc_score(y_eval, Mnom_classifier.predict_proba(X_eval)[:,1]))

    # fd_classifier = open("MultinomialNB.pickle", "wb")
    # fd_vectorizer = open("Vectorizer_tweets.pickle", "wb")
    # pickle.dump(Mnom_classifier, fd_classifier)
    # pickle.dump(vectorizer, fd_vectorizer)
    # fd_classifier.close()
    # fd_vectorizer.close()

    # print("NB Accuracy : ", classify.accuracy(some, ds_eval) * 100) 
    # classifier.show_most_informative_features(15)



    ## TEST ##
    fd_classifier = open("MultinomialNB.pickle", "rb")
    fd_vectorizer = open("Vectorizer_tweets.pickle", "rb")
    MultinomialNB = pickle.load(fd_classifier)
    vectorizer = pickle.load(fd_vectorizer)

    # We scored 80% Accuracy , let's try to get even more !
    print(MultinomialNB.score(vectorizer.transform(ds_tweets.body),y) * 100)


    # test = [("1337 is a coding school for the next generation"), ("ugh i dont feel great arround")]
    # test_vector = vectorizer.transform(test)
    # prediction = MultinomialNB.predict(test_vector)
    # for pred in prediction:
    #     if (pred == 4) :
    #         print(f"Positive :D")
    #     else :
    #         print(f"Negative :C")
    fd_classifier.close()
    fd_vectorizer.close()

    ## MORE TEST's ##

