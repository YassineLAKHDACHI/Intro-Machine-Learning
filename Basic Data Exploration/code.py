# Read the Iowa data file into a Pandas DataFrame called home_data.

import pandas as pd

# Path of the file to read
iowa_file_path = 'C:/Users/Yqssine/Documents/Technocholabes_Internship/Dataset/dataset/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Review The Data
# Print summary statistics in next line
home_data.describe()

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = 10517

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 12
