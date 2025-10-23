## Overview
A credit card company has provided historical data on its active customers. The data includes demographic and behavioral information on customers. Your task is to build a model to predict monthly credit card spend for individual customers.

## Description

#### Variable Description

customer_id: Unique identifier for each customer
age: Age of the customer in years.
gender: Gender of the customer (male, female).
marital_status: Marital status (single, married).
num_children: Number of children in the household.
education_level: Highest educational qualification (high school, bachelors, graduate).
region: Customer’s home region (south, midwest, northeast, west).
employment_status: Employment status (employed, self-employed, unemployed, student).
owns_home: Whether the customer owns their home (1 = yes, 0 = no).
has_auto_loan: Whether the customer has an auto loan (1 = yes, 0 = no).
annual_income: Annual income in dollars.
credit_score: Credit score of the customer (300–850 scale).
num_credit_cards: Number of credit cards the customer holds across all banks.
credit_limit: Credit card limit in dollars.
tenure: Number of years the customer has held the card.
card_type: Type of credit card (standard, gold, platinum).
num_transactions: Number of credit card transactions in the month.
avg_transaction_value: Average dollar value of those transactions.
online_shopping_freq: Number of online purchases made in the month.
reward_points_balance: Total reward points accumulated by the customer.
travel_frequency: Number of trips taken in the past month.
utility_payment_count: Number of utility bill payments made with the card.
monthly_spend (target variable): Total monthly spending on the customer’s credit card in USD.


## Evaluation

#### Goal

Build a predictive model to estimate monthly credit card spending of individual customers based on a rich set of customer attributes, including demographics, credit behavior, transaction activity, and lifestyle indicators.

#### Metric

You must train and validate predictive models that estimate monthly_spend. Submissions will be evaluated based on RMSE (root mean squared error) on the scoring data. Scoring data is split into a public and private dataset. Performance on the public dataset will be posted on Public Leaderboard which will be visible during the competition. Score on the private dataset will be shared on the Private Leaderboard at the conclusion of the competition. Your performance will be judged based on your rank on the Private Leaderboard. Lower your RMSE, higher your rank.

Kaggle allows you to select the submission to use for Private Leaderboard. Unless you are extremely confident in the virtues of a particular submission, it is best to allow Kaggle to automatically use your top Public Leaderboard submission.





