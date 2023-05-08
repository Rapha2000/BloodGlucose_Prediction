# Blood Glucose Level Prediction

---
## 1. Introduction 
---

Diabetes is a chronic disease that affects the body’s ability to regulate blood glucose levels. Glucose is the body’s main source of energy and is regulated by a hormone called insulin, produced by Beta cells from the pancreas. There are two main types of diabetes: type 1 diabetes, which is an autoimmune disease in which the immune system attacks the insulin-producing cells, and type 2 diabetes, which is characterized by insulin resistance and insufficient insulin production by the pancreas. 

As part of a R&D school project in collaboration with a start-up based in Montpellier, France, we worked on a blood glucose prediction problem. Our work is part of a larger project led by the company which aims to develop, in the coming months, a decision support system allowing type 1 diabetics to manage their blood glucose during physical activity. People with type 1 diabetes may have problems with strenuous physical activity. During exercise, the body uses glucose for energy, which can lead to a decrease in blood glucose levels in people with type 1 diabetes. To avoid hypoglycemia during exercise, people with type 1 diabetes should adjust their insulin and carbohydrate intake before and after exercise. However, physical activity doesn’t always induce hypoglycemia, the process is large more harder to understand, as it also depends on the type of physical activity (aerobic, anaerobic, intermediate...) and some others parameters.

This R&D work took place from January to April 2023. At this time we had just finished our generalist course and barely began our course in AI & Data Science. But that was the objective of this work, as fully beginners, we had to introduce ourselves to the world of Data Science through a concrete and real project in order to discover on our own the various issues that a Data Science project can raise. Thus, we were expected to do more research, discovery and autonomous initiation, rather than producing perfect and innovative results. This project really aimed to train us in an autonomous way to diverse and varied notions of AI and Data Science. This is why some of our technical choices or data manipulations which you will be able to see on the notebooks might not be rigorously scientifically correct. The goal of the project was more about make us discover the various issues raised along the process of realizing a Data Science project from scratch.

---
## 2. The Dataset
---

The OhioT1DM Dataset was created to aid research in predicting blood glucose levels. The dataset includes eight weeks of continuous glucose monitoring, insulin, physiological sensor, and self-reported life-event data for 12 people with type 1 diabetes. The OhioT1DM Dataset was first released in 2018 for the inaugural Blood Glucose Level Prediction (BGLP) Challenge, with data for six individuals with type 1 diabetes. The dataset has since (in 2020) been expanded to include an additional six individuals. The authors of this dataset hope that the OhioT1DM Dataset will facilitate research in blood glucose level prediction and lead to improvements in the treatment and management of type 1 diabetes.

The OhioT1DM Dataset comprises eight weeks of data for 12 individuals with type 1 diabetes, who are anonymized using randomly assigned ID numbers. All individuals utilized insulin pump therapy with continuous glucose monitoring (CGM), utilizing Medtronic 530G or 630G insulin pumps and Medtronic Enlite CGM sensors during the data collection period. Life-event data was reported through a custom smartphone app, and physiological data was collected using a fitness band. 

---
## 3. The project
---

You will find 4 files in the repository:

- **details/paper.pdf**: this file documents the whole project, with an introduction about diabetes and then the work we made. This document gather all information useful to understand the project, its context and the work we made

- **details/slides.pdf**: this file are the slides we used during our presentation in front of jury

- **code/pre_processing.ipynb**: in this file we make all the pre-processing work. We have created a function to parse the xml files into csv files to better manipulate our data. We also created a function that builds the dataframe used to train our model.

- **code/TS_analysis.ipynb**: in this file we made some basic analysis specific to time series. 

- **code/BG_forecasting_LSTM.ipynb**: in this file we implement our models: a first univariate model to introduce ourselves to LSTM, a second one with 2 features and then our final multivariate LSTM model using 6 features.

---
## 4. Installation
---

We used Python 3.9.16 and recommand you to create a virtual environment and install the following frameworks versions:

- keras 2.10.0
- pandas 1.5.3
- scikit-learn 1.2.2
- statsmodels 0.13.5
- tensorflow 2.10.0

