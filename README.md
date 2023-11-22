# Brazilian-E-Commerce-Data-Analysis

![Brazilian E-Commerce Data Dashboard](gifdashboard.gif)

[Brazilian E-Commerce Data Dashboard Streamlit App](https://e-commerce-data-analyst.streamlit.app/)

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)

## Overview
This project centers around the analysis and visualization of public e-commerce data. It encompasses code for tasks such as data wrangling, exploratory data analysis (EDA), and the development of an interactive Streamlit dashboard for in-depth exploration of the E-Commerce Public Dataset. The primary goal of this project is to gain insights from the extensive data available in the E-Commerce Public Dataset

## Project Structure
- `dashboard/`: This directory contains dashboard.py which is used to create dashboards of data analysis results.
- `data/`: Contains raw CSV data for data analysis.
- `notebook.ipynb`: This file is used to do data analysis.
- `README.md`: This documentation file.

## Installation
1. Clone this repository to your local machine:
```
git clone https://github.com/RYGcode/Brazilian-E-Commerce-Data-Analysis.git
```
2. Go to the project directory
```
cd Brazilian-E-Commerce-Data-Analysis
```
3. Install the required Python packages by running:
```
pip install -r requirements.txt
```

## Usage
1. **Data Wrangling**: Utilize the data wrangling scripts available in the `notebook.ipynb` file to prepare and clean the dataset.

2. **Exploratory Data Analysis (EDA)**: Dive into the data and gain valuable insights using the provided Python scripts. EDA can enhance your understanding of patterns within the e-commerce public dataset.

3. **Interactive Visualization**: Launch the Streamlit dashboard to interactively explore and visualize the data:

```
cd Brazilian-E-Commerce-Data-Analysis/dashboard
streamlit run dashboard.py
```
Access the dashboard in your web browser at `http://localhost:8501`.

## Data Sources
In this project, the analysis process will be carried out using the Brazilian E-Commerce Public Dataset which can be seen at the [following link](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
