import pandas as pd
import os

# List all CSV files in the directory
csv_files = [file for file in os.listdir('results') if file.endswith('.csv')]

print(csv_files)

# Initialize an empty DataFrame to store combined data
combined_df =  None #pd.DataFrame()

# Iterate through each CSV file
for file in csv_files:
    if 'combined_data' in file:
        continue
    # Read the CSV file
    df = pd.read_csv('results/'+file)
    
    # Extract the file number from the filename
    
    # Rename the 'Label' column to 'Label_fileX'
    df = df.rename(columns={'Label': f'Label_{file[:-4]}'})
    print(f'Label_{file[:-4]}')

    # if combined_df is not None:
    df = df.drop(columns=['Text'])  
    # print(df)
    
    # Merge the current DataFrame with the combined DataFrame based on 'Index', 'Category', 'Text'
    # combined_df = pd.merge(combined_df, df, on=['Index', 'Category', 'Text'], how='outer')
    if combined_df is None:
        combined_df = df
    else:
        # Merge the current DataFrame with the combined DataFrame based on 'Index', 'Category', 'Text'
        # combined_df = pd.merge(combined_df, df, on=['Index', 'Category','Text'], how='outer',)
        # combined_df = pd.merge(combined_df, df, on=['Index', 'Category'], how='outer', suffixes=('', f'_A'))
        combined_df = pd.merge(combined_df, df, on=['Index', 'Category'], how='outer',)

## add text from file with pre processing
df1 = pd.read_csv('/home/dr-nfs/m.badran/mawqif/results/gt.csv')
df1 = df1.drop(df1.columns.difference(['Index','Category','Text']), axis=1)
combined_df = pd.merge(combined_df, df1, on=['Index', 'Category'], how='outer')

print(combined_df.columns)

# Fill NaN values with empty strings
# combined_df = combined_df.fillna('')

# Write the combined DataFrame to a new CSV file
combined_df.to_csv('results/combined_data.csv', index=False)
