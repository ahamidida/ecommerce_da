import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('sample_data.csv')

# Convert the 'date' column to a datetime object
df['date'] = pd.to_datetime(df['date'])

# Format the 'date' column using the strftime() function
df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

df.dropna(inplace=True)

# Save the DataFrame back to a CSV file
df.to_csv('daf.csv', index=False)
