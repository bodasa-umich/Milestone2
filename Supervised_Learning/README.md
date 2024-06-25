**Supervised Learning**

This folder contains the Supervised Learning portion of the project. It is trained on Enron email data that has been labeled by CMU.
We have chosen to build a model that categorizes two kinds of email: General and Collaborative.

The supervised_learning.ipynb is a notebook that contains all the exploratory work and model selection process.
The predict.py is a Python file that can be passed a CSV file as input (a sample file is present in the data/ directory), and outputs the same CSV file with an additional column called 'Prediction'.

The data/ directory also contains a CSV file with the analysis of some wrongly predicted emails.
