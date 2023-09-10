# Grocery Sales Forecasting Challenge for Azubian

[![View Repositories](https://img.shields.io/badge/View-My_Repositories-blue?logo=GitHub)](https://github.com/justinjabo250?tab=repositories)
[![View My Profile](https://img.shields.io/badge/MEDIUM-Article-purple?logo=Medium)](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian/blob/main/Article.md)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-yellow)](https://huggingface.co/spaces/ikoghoemmanuell/SEER-A_sales_forecasting_app)
[![Website](https://img.shields.io/badge/My-Website-darkgreen)](https://emmanuelikogho.netlify.app/)

Increase sales of groceries using exploratory data analysis and machine learning.

![grocery-sales-challenge-readme-azubian-](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian/assets/102419217/88f2040f-72a1-4bb0-936a-f2520ae73a2f)


## Project Overview
In this project, we seek to determine the possibility that a client would leave the business, the primary churn indicators, as well as the retention tactics that may be used to avoid this issue. One of the main issues facing the telecom sector is churn. According to studies, the top 4 wireless providers in the US see an average monthly churn rate of 1.9% to 2%.

One of any company's largest expenses is customer churn. The percentage of consumers who ceased using your company's product or service within a predetermined duration is known as customer churn, also known as customer attrition or customer turnover. For instance, if you started the year with 500 clients and completed it with 480, then 4% of those 500 customers went. It would greatly aid the company in strategizing their retention campaigns if we could determine why a client leaves and when they leave with some degree of accuracy.

## DESCRIPTION
The project aims to analyze and forecast the number of products sold per stores per weeks for a neighborhood grocery store, the goal is to develop a model that can accurately anticipate future sales using data from 54 different stores and 33 different products collected from the same country. In order to assist the management of the business develop inventory and sales plans.

## OBJECTIVE
Using historical sales data from the Grocery sales over the last years have accumulated a lot of data, we will utilize sales forecasting to forecast future sales levels for a company.

Our goal is to assist business managers in forecasting the future using this data, which has been stored on file for a predetermined amount of time after the event or after it occurred.

to build models use those models to make observations, and use them to guide future strategic decision-making. We would like to assist management at the grocery shop in gathering some insights from their data in order to improve operations and eventually revenue.

## Dataset

The Zindi Africa platform makes available the dataset that was used for this case study. It includes information about the product, the store, and historical sales statistics. The dataset is used to create models that can predict future sales with accuracy and assist grocery businesses in streamlining their inventory control and supply chain procedures.


For Azubian on Zindi Africa's Grocery Store Forecasting Challenge, this repository serves as a case study. Predicting the sales of various products at grocery stores using past data is the challenge at hand. The data, methodology, and models examined in this case study are examined to shed light on the predictive analytics procedure.


## Setup

Here is information on Google Colab.
Find out here how to link your github account to Colab.

Note: You may run the notebook on Colab by forking this repository.

## Methodology

1. Exploratory Data Analysis (EDA): The dataset is thoroughly examined at the outset of the case study in order to comprehend its structure, variables, and trends. To get insights into the sales trends and interactions between variables, EDA techniques like data visualization and statistical analysis are used.


2. Feature Engineering: To develop meaningful features that capture pertinent information for sales forecasting, the dataset is preprocessed and modified. In order to account for time dependencies, this entails tasks like resolving missing values, encoding categorical variables, and developing lagged features.


3. Model Development: To find the most effective strategy, many machine learning and time series forecasting models are created and assessed. This could use established regression models, ensemble approaches, or cutting-edge methods created especially for time series forecasting, such ARIMA, SARIMA, or Prophet.


4. Model Evaluation: Using appropriate evaluation metrics like Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE), the performance of the constructed models is evaluated. The models are contrasted based on how well they can predict the future and identify underlying sales trends.


5. Forecasting and Visualization: Sales projections for upcoming time frames are created using the chosen methodology. To offer practical insights and support managerial decision-making, the predicted consequences are displayed using graphs and charts.

## Data

The data set used in this project was sourced from the [Zindi](https://zindi.africa/competitions/grocery-store-forecasting-challenge-for-azubian/data).



## Data set Description

| Column Name         | Type            | Description                                                                             |
|---------------------|-----------------|-----------------------------------------------------------------------------------------|
| Target              | Categorical     | Total sales for a product category at a particular store at a given date                                              |
| Stores_id           | Numeric         | the unique store id                                               |
| Category_id         | Numeric         | the unique Product category id                                                            |
| Date                | Numeric         | date in numerical representation                                    |
| Onpromotion         | Numeric     | gives the total number of items in a Product category that were being promoted at a store at a given date                                            |
| Nbr_of_transactions | Numeric         | The total number of transactions happened at a store at a given date                                         |
| year_weekofyear     | Numeric         | the combination of the year and the week of the year, (year_weekofyear = year*100+week_of_year)                             |
| ID    | Numeric         | the unique identifier for each row in the testing set: year_week_{year_weekofyear}_{store_id}_{Category_id}                                                   |



## Repository Structure

The repository is set up as follows:

- `data/`: This directory contains the dataset files.
- `notebooks/`: Contains Jupyter notebooks that demonstrate the case study's step-by-step implementation, covering feature engineering, EDA, model development, and evaluation.
- `dev/`: This directory contains any source code or scripts used in the case study, such as those for custom functions or data preprocessing.

To better understand the case study process, feel free to peruse the notebooks and source code.


# Project Setup

If you wish to use this app (To run the app), first, Fork this repository. now take the following actions.

1. [`Python 3`](https://www.python.org/) (**a Python version lower than 3.10**) is required to be installed on your system. 

2. When you are at the repository's `root :: repository_name> ...` clone this repository by following the instructions below:



- Windows:

```python
python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt
```

- Linux & MacOs:

```python
python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt
```

Both lengthy command lines follow the same format, pipering multiple commands together with the symbol ";" although you can manually run them in sequence.


1. To prevent conflicts, **Create a virtual Python environment** that isolates the project's necessary libraries;
2. **Switch on (Activate) the Python virtual environment** So that the isolated environment's kernel and libraries are used;
3. **Update Pip, the installed libraries/packages manager** To the most recent release that will function properly;



change directory to where the app is located

```python
cd dev; cd app
```

to run app, install the requirements for the app,

```python
pip install streamlit transformers torch
```

then go to your terminal and run

```python
streamlit run app.py
```

# screenshot

![seer-sales-prediction-for-azubian](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian/assets/102419217/933f5f5b-2976-4499-bd48-050c1bab5bd0)


## 👏 Support

If you found this article helpful, please give it a clap or a star on GitHub!

## Contribution
You contribution, critism etc are welcome. We are willing to colaborate with any data analyst/scientist to improve this project. Thank your 

## Contact

`Justin Jabo` 

`Data Analyst`
`Azubi Africa`

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?logo=linkedin&logoColor=orange)](https://www.linkedin.com/in/jabo-justin-2815341a2/) 

