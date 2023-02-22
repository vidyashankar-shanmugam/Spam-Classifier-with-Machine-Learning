# importing the required libraries
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from imblearn.over_sampling import ADASYN
import wandb
import matplotlib.pyplot as plt


# Creating class for spam classification
class spam_classifier:
    # Initializing the class with dataframe and initialize text normalization methods
    def __init__(self,data,cfg):
        self.df = data
        self.cfg = cfg
        self.norm = hydra.utils.instantiate(cfg.norm_method)

    #Cleaning the data by removing punctuations, numbers and special characters and labelling
    def data_preprocessing(self):
        self.df.columns = ['label','message']
        self.df['message'] = self.df['message'].str.replace('[^a-zA-Z ]',' ', regex = True)
        self.df['message'] = self.df['message'].str.lower()
        self.df['message'] = self.df['message'].str.split()
        le = LabelEncoder()
        self.df['label'] = le.fit_transform(self.df['label'])
        fig = plt.figure(figsize=(10,5))
        plt.bar(self.df['label'].unique(),self.df['label'].value_counts().values)
        plt.show()
        print("Label spam has ",self.df[self.df['label']==1].shape[0]," entries")
        print("Label not spam has ",self.df[self.df['label']==0].shape[0]," entries")
        print("Data is highly imbalanced")

   # Creating corpus by removing stop words and stemming/lemmatizing
    def corpus_creation(self):
        corpus = []
        for i in (self.df['message']):
            if "Lemmatizer" in self.cfg.norm_method._target_:
                temp_list = [self.norm.lemmatize(word) for word in i if not word in set(stopwords.words('english'))]
            else:
                temp_list = [self.norm.stem(word) for word in i if word not in stopwords.words('english')]
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
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.cfg.test_size, random_state=self.cfg.random_state)
        # Balanced the data using Adaptive Synthetic Sampling
        ada = ADASYN(random_state=42)
        X_train, Y_train = ada.fit_resample(X_train, Y_train)
        return X_train, X_test, Y_train, Y_test

    # Training the model with decision tree, support vector machine classifiers
    def model_training(self):
        self.data_preprocessing()
        corpus = self.corpus_creation()
        X_train, X_test, Y_train, Y_test = self.train_test_set_split(corpus)
        classifier = hydra.utils.instantiate(self.cfg.algorithm)
        spamming_detection_model = classifier.fit(X_train, Y_train)
        Y_pred = spamming_detection_model.predict(X_test)
        self.accuracy_metrics_check(Y_pred, Y_test, self.cfg.algorithm._target_.split(".")[-1],
                                    self.cfg.norm_method._target_.split(".")[-1])


    # Computing classification report and confusion matrix metrics
    def accuracy_metrics_check(self, Y_pred, Y_test, classifier, method):
            print(classifier,' - ',method,'\n')
            class_0 = list(classification_report(Y_test, Y_pred, output_dict=True)['0'].values())
            class_1 = list(classification_report(Y_test, Y_pred, output_dict=True)['1'].values())
            var = {x[0]: x[1] for x in
                   zip(["Method", "Classifier", "0 Precision", "0 Recall", "0 F1-Score", "1 Precision", "1 Recall",
                        "1 F1-Score"],
                       [method, classifier, class_0[0], class_0[1], class_0[2], class_1[0], class_1[1], class_1[2]])}
            wandb.log(var)
            wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=Y_test, preds=Y_pred)})
            print(classification_report(Y_pred,Y_test,output_dict=True),'\n')
            print('Confusion matrix', '\n',confusion_matrix(Y_pred,Y_test),'\n')

@hydra.main(config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="spam-classification", config=config_dict)
    df = pd.read_csv('email_data.csv')
    df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
    spam_obj = spam_classifier(df, cfg)
    spam_obj.model_training()
    wandb.finish()

if __name__ == "__main__":
    my_app()