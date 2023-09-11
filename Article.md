# Focast Grocery Sales Using Machine Learning 
![ride-the-cart-at-the-market](https://github.com/justinjabo250/Sales-Analysis-and-Forecasting-Of-The-Grocery-Stores/assets/115732734/7bff8484-afb0-4581-8730-dd45ea242d63)
 


Azubian, a firm with grocery stores all over Africa, has recently experienced a fall in revenue, mostly as a result of losing clients as a result of understocking in-demand items. Customers may become dissatisfied and purchase elsewhere if they visit their stores and discover that basic commodities like rice are not available. Their inability to store all of their product has proven to be another problem.

Azubian needs the assistance of a professional data analyst like you/me to handle these problems and enhance their company picture. They have asked you to review their sales data and make insightful suggestions for the future. With the help of your study, they will be able to properly manage their inventory and anticipate client demand by identifying the products that are most likely to run out of stock and forecasting the quantities needed. 


## What is time series analysis?


A time series regression problem exists here. Let me explain that now. What is time series analysis, to begin with?

Time is a factor in time series analysis, so you must examine a dataset with a date column. 

A cross-sectional dataset is acquired at a fixed moment for a dataset where time doesn't matter, but a time series state is a data collection recorded at successive periods in time. 

You should be able to find out how much sleep you received over the course of the month, for instance, if your smartwatch tracks your sleep. My sleep patterns are depicted in the graph below over a certain time. My sleep patterns are displayed on the y-axis, and the month's day is displayed on the x-axis. Therefore, the dataset transforms into a Time Series dataset when time is involved.


![my sleep chart](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689435953689_image.png)


Watch [this video](https://youtu.be/FsroWpkUuYI) to know more about this.


## Importance of a Time Series Data Project


Why is this crucial and important? 

This is so because time has an impact on the majority of enterprises nowadays. Sales of a product and revenue of a firm are typically impacted by time. For instance, during the Christmas season, clothing sales are at their peak. Agriculture is yet another example where time series analysis may be used to predict the ideal time of year to plant crops.

How do you tell if a dataset contains time series data? It's straightforward: if a time variable is present (often the index), the dataset is a time series dataset; if not, it is not. 

![grcery](https://github.com/justinjabo250/Sales-Analysis-and-Forecasting-Of-The-Grocery-Stores/assets/115732734/6cfd92a2-4d1f-42f3-ac03-8dfaeb99e7a1)

## Aim of the project


In this research Project, we hope to forecast the sales of goods found in Azubian's grocery stores. Which products sell better and when do we want to know? In order to prevent overstocking or understocking, we also wish to forecast future sales. 

For instance, we would like to know how much chicken to purchase for the upcoming eight weeks as well as how many tomatoes will be sold during that time. 

If we overstock such perishable commodities, they would wind up spoiling, which is a loss. By always having what customers want to buy available when they need it, we hope to please customers. This might provide our grocery retailers with a market advantage today.


We would have created a machine learning model to anticipate product sales pay week over the upcoming eight weeks by the time this project was finished. For each store in each location, we want to obtain this data.

Finally, we'll use Streamlit to deploy this model as a web application.



Clone the repo and get started from  [here.](https://github.com/justinjabo250/Sales-Analysis-and-Forecasting-Of-The-Grocery-Stores)

# Dataset description

The dataset made available for this research is divided into 6 files and spans 4 years. Here is a quick summary of the dataset.

**train.csv**

The model will be trained using the data in this file. It includes the sales-focused target column. Another column in it is named "onpromotion." This column will inform us of the number of products that were being advertised (promoted) at the time.



![Preview of train.csv file](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689421639944_image.png)


**holidays.csv**

The information in this file relates to holidays.


![preview of holidays.csv](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689421953863_image.png)


**test.csv**

Similar to train.csv file, but lacking the target column. This file will be used to assess how well the trained model performed.


![Preview of test.csv file](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689422068460_image.png)


**stores.csv**

Contains details on each store, including its city, type, and cluster.


![Preview of stores.csv file](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689422175669_image.png)


**dates.csv**

Describes each day of each year in our collection, along with the date attributes that go with it.


![Preview of dates.csv file](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689422552387_image.png)



## Issues with the data

Here are the main problems that were discovered while previewing the dataset.


![Issues with the data and how to solve them](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689423176844_image.png)




# Analysis and cleaning of the dataset.

Let's begin prepping the dataset for analysis at this point.


## Observations

Notably, for the purposes of this post, I won't include them all. Here are some of my main findings.


- The dates' date column.The dates for the train are covered by the csv file, which spans from 365 to 1684.test.csv and csv files.
- No null values are present.
- There are no values that are missing.
- Aside from the holidays.csv file, no other csv file contains a holiday column.


## Data cleaning

Let me briefly describe the main actions I performed to prepare the dataset.

1. I combined the year, month, and day columns of the dates to get a valid date column.entire date column in a csv file. The date index for the other files will be this column.



    def get_datetime(df):
      # Create a new column combining the year, month, and day of the month in the desired format (yyyy-mm-dd)
      df['date_extracted'] = (
          dates['year'].astype(int).add(2000).astype(str) + '-' +
          dates['month'].astype(str).str.zfill(2) + '-' +
          dates['dayofmonth'].astype(str).str.zfill(2)
      )
    
    get_datetime(dates)



2. I applied similar reasoning to the holidays.csv file and I created a column called “holiday_type". It is a holiday if the date appears in the holidays.csv file. If not, that day is a workday.



    def get_is_holiday_column(df):
      df['holiday_type'] = df['holiday_type'].fillna('Workday')
    
      # create column to show if its a holiday or not (non-holidays are zeros)
      df['is_holiday'] = df['holiday_type'].apply(
          lambda x: False if x=='Workday'
          else True)



3. Set the newly created date column as the index.


    def set_index(df):
      df.drop('date', inplace=True, axis=1)
      df.set_index('date_extracted', inplace=True)
    set_index(train_merged2)
    set_index(test_merged2)



4. Only "2003-02-29" was found to be an invalid date, so before converting to datetime, we shall set invalid dates to NaT. Then enter "2003-02-29," which is a leap year's February 29.



    train_merged2['date_extracted'] = pd.to_datetime(train_merged2['date_extracted'], errors='coerce')
    test_merged2['date_extracted'] = pd.to_datetime(test_merged2['date_extracted'])
    train_merged2['date_extracted'].fillna('2003-02-29')



# Exploratory data analysis

We'll pose some queries (questions) and formulate a hypothesis in this part.


## Hypothesis

***Null Hypothesis***: holidays have a big effect on sales, hence the sales data is seasonal.
***Alternative Hypothesis***: holidays don't affect sales, hence sales data is stationary.

## Hypothesis Validation

Let’s check if our null hypothesis is true or not.

![Bar chart of sales vs holiday type](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689428352904_image.png)

![Box plot of sales during holidays vs non-holidays](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689428466937_image.png)


The hypothesis H1, which states that holidays affect sales and the sales data is seasonal, is more likely to be true. Alternative Hypothesis REJECTED!


## Questions & answers  
1. **How do sales vary by promotion status?**


![](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689429178014_image.png)



We can see from the plot above that promotions have a significant impact on overall sales. Sales are higher when a product is on promotion.



2. **Is there a relationship between sales and** **transactions?**
![Scatterplot of sales vs transactions](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689429374239_image.png)


Sales are not affected by transactions. The aprevious graphic results demonstrates that there is little correlation (no link) between sales and transactions, mean "no correlation at all".



3. **How does sales vary during holidays compared to non-holidays?**
![Bar chart and box plot of sales during holidays vs non-holidays](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689429618307_image.png)


Sales during holidays are higher than sales during non-holidays, as shown in the chart on the left. Why does this matter? Holiday sales alone generated more revenue than all other days of the week put together. The right-hand graphic demonstrates that the outliers on holidays are significantly greater than those on non-holiday days. This is anticipated because the peak sales periods are often around the holidays.


4. **What is the trend of sales over time?** 
![plot of sales over time](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689430307584_image.png)


The aprevious results (the above chart) data shows that sales have been rising (increasing) consistently over time.


5. **How much does promotion affect sales of different product categories?**
![](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689433239456_InkedgG9Q9kfT_LI.jpg)


From the above, it is clear that some product categories are less impacted by promotions than others. For instance, 'category 0' on the left shows higher sales with more advertising than 'category 7' on the right.


6. **Is our sales data seasonal?**
![monthly sales over time](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689431531919_image.png)

> What is the difference between seasonality and trend? Seasonality and trend are two different concepts in time series analysis. Seasonality refers to the repeating patterns in a time series that occur over a fixed period of time, such as daily, weekly, or monthly. Trend refers to the long-term direction of a time series, such as increasing or decreasing.
> 
> For example, the number of ice cream sales might be seasonal, with higher sales in the summer and lower sales in the winter. The number of people using a social media platform might have a trend, with increasing usage over time.


It might be difficult to determine seasonality by simply looking at a chart. So let's check it using a statistical method. The KPSS examination will be used.


    # Assuming your time series data is stored in the variable 'sales_data'
    sales_data = train['target']


    # Perform KPSS test
    kpss_result = kpss(sales_data)
    kpss_statistic = kpss_result[0]
    kpss_pvalue = kpss_result[1]
    kpss_critical_values = kpss_result[3]


![](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689432517211_image.png)


If p-value is greater than 0.05 (p-value > 0.05), sales data is stationary. Since 0.01 < 0.05, This means that our sales is non-seasonal and there is no repeating pattern over time.


![birds-of-prey](https://github.com/justinjabo250/Sales-Analysis-and-Forecasting-Of-The-Grocery-Stores/assets/115732734/24a6fbbb-52b8-4ff5-9e10-41db24382de2)


# Recommendations 

Giving stakeholders practical recommendations is the aim of data analysis. The company should make the following data-driven decisions to increase sales.


> **Utilize promotions**: Promotions have a significant impact on sales. Consider implementing strategic promotional campaigns for products to boost sales. Identify products that have higher sales when on promotion and focus on promoting them more frequently.
> 
> **Focus on sales drivers**: While transactions do not seem to directly affect sales, it’s important to identify other factors that drive sales. Analyze customer behavior, product attributes, pricing, and other variables to understand what influences sales the most. Use this information to optimize your sales strategies.
> 
> **Capitalize on holidays**: Holidays drive higher sales compared to non-holidays. Plan and execute targeted marketing and promotional campaigns during holiday seasons to take advantage of increased consumer spending. Ensure sufficient inventory and resources to accommodate the surge in demand during these periods.
> 
> **Monitor and adapt to sales trends**: Sales have been consistently increasing over time. Continuously monitor sales data and identify emerging trends. Adjust your inventory, marketing, and sales strategies accordingly to maximize the opportunities presented by the growing sales trend.
> 
> **Tailor promotions to product categories**: Different product categories respond differently to promotions. Analyze the impact of promotions on various product categories and allocate promotional efforts accordingly. Focus more on categories that show a higher increase in sales with promotions to maximize their potential.
> 
> **Leverage non-seasonality**: Since the sales data is non-seasonal, explore other factors driving customer demand and purchasing behavior. Consider conducting customer surveys, analyzing market trends, and studying competitors to identify opportunities for growth outside of seasonal patterns.


Now that you are familiar with the dataset and have heard some of my insights, It's time to begin machine learning modeling. Hurray!

# Machine learning Modelling

We can train two different types of models to solve a time series regression problem. The conventional time series models, such as the AR model, are available. Then there are the machine learning models that you are familiar with, such as linear regression. They both have benefits and drawbacks. Let Me Describe.

Traditional time series models employ correlation to forecast the future using data from the past. They rely on the relationship between current values and those of the future. In this instance, they would rely on the possibility that, for instance, the sales from last week might be related to or correlate with the sales from this week.

The partial autocorrelation of the sales column alone is depicted in the diagram below. The 3 lines highlighted indicate that, because this is a time series project, each sales value correlates with the previous three sales values. It's autocorrelation, then. The traditional time series models operate on this principle. It's interesting, 🤯 I know.


![Plot of partial autocorrelation](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689436910282_Inked_-ggIk3B_LI.jpg)


A feature-driven strategy is the use of machine learning models. These models can accept a variety of attributes as input data. Choose the aspects that are most crucial for your training while using this strategy. 

A basic comparison of the two is provided in the table below.


![machine learning vs traditional time series models for time series modelling](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689437265730_image.png)


This is our outcome following training using the two methods.



![traditional time series models](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689437759039_image.png)



![machine learning models](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689437748585_image.png)


I chose the KNN model from the second technique, which has an RMSE score of 123.11, as the model I would use to develop the app. I chose a model from the second method since it had more features that the user could experiment with during training.


# Model deployment 

The project's simplest component is this. After finishing the data analysis and training the model, we can now deploy the top-performing model as a user friendly web application.

I used Pickle to export the best model and encoder from the notebook, then Streamlit to create a web application. A Python framework called Streamlit makes it simple to launch web apps quickly. Let me walk you through step-by-step all the code I used to create the app. [Try the app.](https://huggingface.co/spaces/ikoghoemmanuell/SEER-A_sales_forecasting_app)


![screenshot of the app](https://github.com/ikoghoemmanuell/Grocery-Store-Forecasting-Challenge-For-Azubian/raw/main/seer-sales-prediction-for-azubian.gif)



## Importing necessary requirements


    import streamlit as st
    import pandas as pd
    import numpy as np
    from PIL import Image
    import requests
    from bokeh.plotting import figure
    from bokeh.models import HoverTool
    import joblib
    import os
    from date_features import getDateFeatures

We first import the necessary libraries, such as streamlit, pandas, numpy, and similar ones. We also imported a method from the *date_features.py* section of Python. The function *getDateFeatures* in this file returns date features such the year, month, day of the month, week of the year, day of the week, season, and so on. It accepts a dataframe as input.



    import numpy as np
    
    # Define the getDateFeatures() function
    def getDateFeatures(df):
        df['holiday_type'] = 'Workday'
        df['is_holiday'] = False
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['dayofmonth'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['weekofyear'] = df['date'].dt.weekofyear
    
        df['quarter'] = df['date'].dt.quarter
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
        # Extract the 'year' and 'weekofyear' components from the 'date' column
        df['year_weekofyear'] = df['date'].dt.year * 100 + df['date'].dt.weekofyear
    
        # create new coolumns to represent the cyclic nature of a year
        df['dayofyear'] = df['date'].dt.dayofyear
        df["sin(dayofyear)"] = np.sin(df["dayofyear"])
        df["cos(dayofyear)"] = np.cos(df["dayofyear"])
    
        df["is_weekend"] = np.where(df['dayofweek'] > 4, 1, 0)
    
        # Define the criteria for each season
        seasons = {'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Autumn': [9, 10, 11]}
    
        # Create the 'season' column based on the 'date' column
        df['season'] = df['month'].map({month: season for season, months in seasons.items() for month in months})
        
    
        return df

The *app.py* file and the file containing this function must both be in the same directory. 



## Loading machine learning assets

The model and encoder files, which must be in the same directory as the *app.py* file, are now loaded for machine learning. 


    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the model from the pickle file
    model_path = os.path.join(current_dir, 'model.pkl')
    model = joblib.load(model_path)
    
    # Load the scaler from the pickle file
    encoder_path = os.path.join(current_dir, 'encoder.pkl')
    encoder = joblib.load(encoder_path)



# Building the interface
## Basic configurations

The app's page configurations, including the title, layout, initial side bar state, and page icon, must first be configured.



    # Set Page Configurations
    st.set_page_config(page_title="Sales Prediction App", page_icon="fas fa-chart-line", layout="wide", initial_sidebar_state="auto")

We are currently setting up the app's sidebar. Within the program, the user ought to have a choice between two pages. both the about page and the homepage.


![code for the sidebar](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689440921302_image.png)



## Home section

You will first be introduced to the app if you are in the home section, and you will then be asked to fill out an input form and supply certain parameters for sales forecast. 



![Intro part of the home page](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689441116579_image.png)


The start date, end date, onpromotion, and product category should all be in column one.

The information about the store should be in column 2, along with the city and cluster, the store type, and the store ID.


![input form on the home page](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689441304475_image.png)


All of this data will now be saved as a dataframe.



    predicted_data = pd.DataFrame(columns=['Start Date', 'End Date', 'Store', 'Category', 'On Promotion', 'City', 'Cluster', 'Predicted Sales'])

We are almost finished building the homepage right now. There are now only two buttons left. both the "clear data" and "predict" buttons.

The predict function is used by the predict button. The dataframe is supplied into this function, which then conducts the encoding and utilizes the model to forecast sales. A value error stating "No sales data provided" will be displayed if the sales data is empty. Otherwise, it will yield the anticipated sales.



![predict function](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689441600975_image.png)


**Predict button**

The start date and end date are checked if you click the predict button; if the start date occurs after the end date, an error is raised that reads, "Start date should be earlier than the end date." If not, it will say "Total sales for the period is: #salesValue" instead of the expected sales.


    
        if st.button('Predict'):
            if start_date > end_date:
                st.error("Start date should be earlier than the end date.")
            else:
                with st.spinner('Predicting sales...'):
                    sales_data = pd.DataFrame({
                        'date': pd.date_range(start=start_date, end=end_date),
                        'store_id': [selected_store] * len(pd.date_range(start=start_date, end=end_date)),
                        'category_id': [selected_category] * len(pd.date_range(start=start_date, end=end_date)),
                        'onpromotion': [onpromotion] * len(pd.date_range(start=start_date, end=end_date)),
                        'city': [selected_city] * len(pd.date_range(start=start_date, end=end_date)),
                        'store_type': [selected_store1] * len(pd.date_range(start=start_date, end=end_date)),
                        'cluster': [selected_cluster] * len(pd.date_range(start=start_date, end=end_date))
                    })
                    try:
                        sales = predict(sales_data)
                        formatted_sales = round(sales[0], 2)
                        predicted_data = predicted_data.append({
                             'Start Date': start_date,
                             'End Date': end_date,
                             'Store': selected_store,
                             'Category': selected_category,
                             'On Promotion': onpromotion,
                             'City': selected_city,
                             'Cluster': selected_cluster,
                             'Predicted Sales': formatted_sales}, ignore_index=True)
    
                        st.success(f"Total sales for the period is: #{formatted_sales}")
                    except ValueError as e:
                        st.error(str(e))
                        


**Clear data button**

When you click the "clear data" button, the dataframe is reset to zero and you get the message "Data cleared successfully."


                        
        if st.button('Clear Data'):
            predicted_data = pd.DataFrame(columns=['Start Date', 'End Date', 'Store', 'Category', 'On Promotion', 'City', 'Cluster', 'Predicted Sales'])
            st.success("Data cleared successfully.")



## About section

You can learn more about the app's purpose from the about section. You have a logo for the app and some text that describes its functions.


![about section code](https://paper-attachments.dropboxusercontent.com/s_DC0D7E849EF6E4D29DB76BE55996037A4C6225ACFEF58DF9AABB10241031FC0B_1689442553146_image.png)



## Hosting the app


You can host your app using a service called Hugging Face to make it available online. To help you through the procedure, they offer a [tutorial](https://huggingface.co/docs/hub/spaces-overview). It's an excellent resource for teaching you how to perform anything step-by-step.



# Conclusion 

Using Streamlit, we have performed exploratory data analysis on shop data and created a machine-learning model that forecasts future sales. The best-performing model has also been made available as a user-friendly online application. 


This project's source code is accessible [here.](https://github.com/justinjabo250/Sales-Analysis-and-Forecasting-Of-The-Grocery-Stores)

Kindly, If you enjoyed reading this article, please click "like," "clap," or leave a comment. 


## Resources
- [What is time series data?](https://youtu.be/FsroWpkUuYI)
- [What is time series analysis?](https://youtu.be/GE3JOFwTWVM)
- [Get started with Streamlit](https://docs.streamlit.io/library/get-started/create-an-app)
- [A step by step tutorial playlist](https://www.youtube.com/playlist?list=PLa6CNrvKM5QU7AjAS90zCMIwi9RTFNIIW)
- [Huggingface Spaces tutorial](https://huggingface.co/docs/hub/spaces-overview)
- [Showcase your model demos with 🤗 Spaces](https://www.youtube.com/watch?v=vbaKOa4UXoM)

