# Spam Classifier

This program is a spam classification tool that uses machine learning algorithms to detect spam emails.
The program takes in a dataset of emails and uses a variety of text normalization and machine learning
techniques to classify emails as spam or not spam.

#### Installation

The program requires the following packages to be installed:

        * hydra
        * wandb
        * pandas
        * nltk
        * scikit-learn
        * imblearn

#### Usage

The program takes in a configuration file located in the conf directory, which specifies the machine learning algorithm, the text normalization method, and the test size. The default configuration can be found in
conf/config.yaml. You can modify the configuration file to change the settings of the program.

The program reads in a dataset of emails. You can modify the file path to point to your own dataset. I have compared the performance of the three text normalization methods as follows:

        *Stemmer
        *SnowballStemmer
        *Lemmatizer

The program outputs the accuracy metrics of the trained model, including a classification report and a
confusion matrix. These metrics are also logged to Weights and Biases for further analysis.