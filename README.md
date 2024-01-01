# Research_and_dev_text_revenue_prediction
This research leverages advanced natural language processing techniques to analyze the textual content of company 10-K and 10-Q filings for the purpose of predicting future revenue. The study employs two key text to vector representation: Term Frequency-Inverse Document Frequency (TF-IDF) analysis and ChatGPT embeddings. The findings contribute to the growing field of financial forecasting by demonstrating the effectiveness of combining traditional statistical methods with state-of-the-art natural language processing approaches for enhanced predictive analytics in the quantative finance domain.


#### [Short Report of the Project](report.pdf)

# Jupyter Notebooks Included in the Repository
[Preprocessing.ipynb](Preprocessing.ipynb): jupyter notebook with preprocessing of the 10k-10q filings of publically traded companies between 2010 and 2021.
- We have utilized spacy library with "en_core_web_lg" pipeline to break the filiing into list of sentences.
- We checked if the sentences have any keywords we have selected as terms signaling R&D to curate our corpus.

[Labels_Creator.ipynb](Labels_Creator.ipynb): jupyter notebook that creates labels for the companies based on CAGR of revenue growth from a base year 
(CAGR Rev > Q1: 1,  Q1 < CAGR Rev < Q3: 2, CAGR Rev > Q3: 3)
- We have utilized Financial Modeling Prep API for revenue data.

[TFIDF_Regressions.ipynb](TFIDF_Regressions.ipynb): jupyter notebook with logistic regression/svm for different years of TFIDF representation of the filings against quartile based labels of R&D Growth.
- collected following metrics for Logistic Regression and SVM:
"lr_train_acc": "Ratio of correctly predicted instances by a logistic regression model on the training dataset",

"lr_test_acc": "Ratio of correctly predicted instances by a logistic regression model on the testing dataset",

"svc_train_acc": "Ratio of correctly predicted instances by a Support Vector Classifier (SVC) model on the training dataset",

"svc_test_acc": "Ratio of correctly predicted instances by an SVC model on the testing dataset",

"lr_precision": "Ratio of true positives to the sum of true positives and false positives in logistic regression",

"lr_recall": "Ratio of true positives to the sum of true positives and false negatives in logistic regression",

"lr_f1": "Harmonic mean of precision and recall in logistic regression",

"svc_precision": "Ratio of true positives to the sum of true positives and false positives in a Support Vector Classifier (SVC) model",

"svc_recall": "Ratio of true positives to the sum of true positives and false negatives in an SVC model",

"svc_f1": "Harmonic mean of precision and recall in a Support Vector Classifier (SVC) model",

"OVO_auroc": "Measure of the ability of a classifier to distinguish between classes using the One-vs-One strategy in multi-class classification",

"OVR_auroc": "Measure of the ability of a classifier to distinguish between classes using the One-vs-Rest strategy in multi-class classification"


Findings: Logistic Regression out performed SVM in out of sample test, Strong AUROC for the regression

[GPT_embeddings_Regressions.ipynb](GPT_embeddings_Regressions.ipynb): jupyter notebook with logistic regression/svm for different years of GPT_embeddings of the filings against quartile based labels of R&D Growth.
- collected same metrics as with TDIFD_Regressions

Findings: Logistic Regression still out performed SVM in out of sample test, Accuracy and AUROC for GPT_embeddings were slightly better but not siginifcant.

[Backpropped_Regressions.ipynb](Backpropped_Regression.ipynb): jupyter notebook with logistic regression/svm for different years of backpropped_embeddings of the filings against quartile based labels of R&D Growth.
- collected same metrics as with TDIFD_Regressions

Findings: Contrary to our hypothesis, the model were significantly way worse. Out of sample accuracy of 

[Cross_year_check.ipynb](Cross_Year_Check.ipynb): jupyter notebook that tests the logistic regression/svm of a given year using data from years in the future to test robustness of the models.

Findings: Strong AUVROC for shorter lag (1, 2), gets weaker with increase in lags(4, 5). OVO_auroc range for t+1: [0.689, 0.742], t+5: [0.655, 0.710]

Stocks_Returns_Collector.ipynb: jupyter notebook to collect stock returns to see how stocks selected bu our model perform in comparison to an benchmark.
Findings: (work in progress), pickle file for renevnue got corrupted, collecting for the 3rd time.

# Folder Included in the Repository 





