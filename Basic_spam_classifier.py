## importing the relevant libraries
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from imblearn.over_sampling import ADASYN
import wandb


# Creating class for spam classification
class spam_classifier:
    # Initializing the class with dataframe and initialize text normalization methods
    def __init__(self,data):
        self.df = data
        self.stemmer = PorterStemmer()
        self.snowball = SnowballStemmer('english', ignore_stopwords=True)
        self.lemmatizer = WordNetLemmatizer()

    #Cleaning the data by removing punctuations, numbers and special characters and labelling
    def data_preprocessing(self):
        self.df.columns = ['label','message']
        self.df['message'] = self.df['message'].str.replace('[^a-zA-Z ]',' ', regex = True)
        self.df['message'] = self.df['message'].str.lower()
        self.df['message'] = self.df['message'].str.split()
        le = LabelEncoder()
        self.df['label'] = le.fit_transform(self.df['label'])
        print("Label spam has ",self.df[self.df['label']==1].shape[0]," entries")
        print("Label ham has ",self.df[self.df['label']==0].shape[0]," entries")
        print("Data is highly imbalanced")

   # Creating corpus by removing stop words and stemming/lemmatizing
    def corpus_creation(self,method):
        corpus = []
        for i in (self.df['message']):
            if method == 'stemmer':
                temp_list = [self.stemmer.stem(word) for word in i if word not in stopwords.words('english')]
                temp_list = ' '.join(temp_list)
                corpus.append(temp_list)
            elif method == 'snowball':
                temp_list = [self.snowball.stem(word) for word in i]
                temp_list = ' '.join(temp_list)
                corpus.append(temp_list)
            elif method == 'lemmatizer':
                temp_list = [self.lemmatizer.lemmatize(word) for word in i if word not in stopwords.words('english')]
                temp_list = ' '.join(temp_list)
                corpus.append(temp_list)
        return corpus

    # Using textual data into a numerical matrix by TF-IDF vectorization
    def text_vectorization(self, corpus):
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(corpus).toarray()
        Y = self.df['label'].values.flatten()
        return X,Y

    # Splitting dataset into training and test sets
    def train_test_set_split(self, corpus):
        X,Y = self.text_vectorization(corpus)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        # Balanced the data using Adaptive Synthetic Sampling
        ada = ADASYN(random_state=42)
        X_train, Y_train = ada.fit_resample(X_train, Y_train)
        return X_train, X_test, Y_train, Y_test

    # Training the model with decision tree, support vector machine classifiers
    def model_training(self):
        self.data_preprocessing()
        methods = ['stemmer','snowball','lemmatizer']
        DT = DecisionTreeClassifier()
        SVM = SVC()
        classifiers = [DT, SVM]
        Classifiers = ['Decision tree classifier', 'Support Vector Machine']
        predictions = []
        for method in methods:
            corpus = self.corpus_creation(method)
            X_train, X_test, Y_train, Y_test = self.train_test_set_split(corpus)
            for classifier in classifiers:
                spamming_detection_model = classifier.fit(X_train, Y_train)
                Y_pred = spamming_detection_model.predict(X_test)
                predictions.append((method, Y_pred ,Classifiers[classifiers.index(classifier)]))
                self.accuracy_metrics_check(Y_pred, Y_test, Classifiers[classifiers.index(classifier)], method)


    # Computing classification report and confusion matrix metrics
    def accuracy_metrics_check(self, Y_pred, Y_test, classifier, method):
            print(classifier,' - ',method,'\n')
            print(classification_report(Y_pred,Y_test),'\n')
            print('Confusion matrix', '\n',confusion_matrix(Y_pred,Y_test),'\n')

if __name__=='__main__':
    df = pd.read_csv('/mnt/c/Users/Admin/OneDrive/Desktop/Projects/spam classifier/email_data.csv')
    df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
    obj = spam_classifier(df)
    obj.model_training()## importing the relevant libraries
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from imblearn.over_sampling import ADASYN
import wandb


# Creating class for spam classification
class spam_classifier:
    # Initializing the class with dataframe and initialize text normalization methods
    def __init__(self,data):
        self.df = data
        self.stemmer = PorterStemmer()
        self.snowball = SnowballStemmer('english', ignore_stopwords=True)
        self.lemmatizer = WordNetLemmatizer()

    #Cleaning the data by removing punctuations, numbers and special characters and labelling
    def data_preprocessing(self):
        self.df.columns = ['label','message']
        self.df['message'] = self.df['message'].str.replace('[^a-zA-Z ]',' ', regex = True)
        self.df['message'] = self.df['message'].str.lower()
        self.df['message'] = self.df['message'].str.split()
        le = LabelEncoder()
        self.df['label'] = le.fit_transform(self.df['label'])
        print("Label spam has ",self.df[self.df['label']==1].shape[0]," entries")
        print("Label ham has ",self.df[self.df['label']==0].shape[0]," entries")
        print("Data is highly imbalanced")

   # Creating corpus by removing stop words and stemming/lemmatizing
    def corpus_creation(self,method):
        corpus = []
        for i in (self.df['message']):
            if method == 'stemmer':
                temp_list = [self.stemmer.stem(word) for word in i if word not in stopwords.words('english')]
                temp_list = ' '.join(temp_list)
                corpus.append(temp_list)
            elif method == 'snowball':
                temp_list = [self.snowball.stem(word) for word in i]
                temp_list = ' '.join(temp_list)
                corpus.append(temp_list)
            elif method == 'lemmatizer':
                temp_list = [self.lemmatizer.lemmatize(word) for word in i if word not in stopwords.words('english')]
                temp_list = ' '.join(temp_list)
                corpus.append(temp_list)
        return corpus

    # Using textual data into a numerical matrix by TF-IDF vectorization
    def text_vectorization(self, corpus):
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(corpus).toarray()
        Y = df['label'].values.flatten()
        return X,Y

    # Splitting dataset into training and test sets
    def train_test_set_split(self, corpus):
        X,Y = self.text_vectorization(corpus)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        # Balanced the data using Adaptive Synthetic Sampling
        ada = ADASYN(random_state=42)
        X_train, Y_train = ada.fit_resample(X_train, Y_train)
        return X_train, X_test, Y_train, Y_test

    # Training the model with decision tree, support vector machine classifiers
    def model_training(self):
        self.data_preprocessing()
        methods = ['stemmer','snowball','lemmatizer']
        DT = DecisionTreeClassifier()
        SVM = SVC()
        classifiers = [DT, SVM]
        Classifiers = ['Decision tree classifier', 'Support Vector Machine']
        predictions = []
        for method in methods:
            corpus = self.corpus_creation(method)
            X_train, X_test, Y_train, Y_test = self.train_test_set_split(corpus)
            for classifier in classifiers:
                spamming_detection_model = classifier.fit(X_train, Y_train)
                Y_pred = spamming_detection_model.predict(X_test)
                predictions.append((method, Y_pred ,Classifiers[classifiers.index(classifier)]))
                self.accuracy_metrics_check(Y_pred, Y_test, Classifiers[classifiers.index(classifier)], method)


    # Computing classification report and confusion matrix metrics
    def accuracy_metrics_check(self, Y_pred, Y_test, classifier, method):
            print(classifier,' - ',method,'\n')
            print(classification_report(Y_pred,Y_test),'\n')
            print('Confusion matrix', '\n',confusion_matrix(Y_pred,Y_test),'\n')

if __name__=='__main__':
    df = pd.read_csv('/mnt/c/Users/Admin/OneDrive/Desktop/Projects/spam classifier/email_data.csv')
    df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
    obj = spam_classifier(df)
    obj.model_training()