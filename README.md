# Semantic Exploration & Retrieval-Augmented Generation over Enron Data

Our goal was to explore machine learning techniques for identifying & suggesting actions based on email content, relevant to legal or customer support scenarios. Given the challenge of accessing real email & supporting manual datasets, we used Enron documents as a proxy for a knowledge base & the email corpus as a stand-in for customer interactions. We aimed to evaluate & identify question topics that could come via any channel, including chatbots. Our unsupervised topic exploration & supervised email content identification simulate tasks data scientists might perform in legal or customer operations.
The SEC documents provide a basis for topic formation related to the Enron scandal, while the emails contain both general content & topics covered in the case documents, although imbalanced. Stakeholders include managers & customers in legal & support contexts.
Our supervised learning focuses on categorizing email content, while our unsupervised scope explores topic identification & modeling. We also consider retrieval augmented generation (RAG) as a future extension for unsupervised learning, leaving detailed evaluation for subsequent projects.

## Repository Structure

- We have 3 parts in this repository: Supervised Learning, Unsupervised Learning and RAG (all directories prefixed by RAG_). Each directory is self contained, i.e., it contains a folder called data/
- Our exploratory and explanatory code lies in ipynb files within each directory.
- The RAG Demo App is also accessible through the link here: https://huggingface.co/spaces/gschaumb/M2-Team01-SS24-RAG-demo
- Our final project report is located in the main directory, titled 'Milestone 2 Project Report.pdf'
