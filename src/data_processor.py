import pandas as pd
from matplotlib import pyplot as plt


class DataProcessor:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        print(self.data.head())
        # Descriptive Statistics for numerical data (prices, discount percentage, rating):
        print(self.data[['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count']].describe())

        # Convert 'rating' to a numeric type, coercing any errors
        self.data['rating'] = pd.to_numeric(self.data['rating'], errors='coerce')

        # Distribution of Ratings and Discounts to understand customer preferences and sales patterns:
        # Histogram of ratings
        self.data['rating'].plot(kind='hist', title='Rating Distribution')
        plt.show()

        # Histogram of discount percentages
        self.data['discount_percentage'].plot(kind='hist', title='Discount Percentage Distribution')
        plt.show()

    def preprocess(self):
        # Preprocess the data (e.g., handling missing values, normalization)
        pass

    # Additional methods to retrieve data as needed for the chatbot
