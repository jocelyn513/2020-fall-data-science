import pandas as pd
import numpy as np

# Load data here
df = pd.read_csv('data/listings.csv', sep=',')
df

# 1. How many listings are there with a price less than 100?
condition_1 = df['price'] < 100

len(df[condition_1].id.value_counts())

# 2. Make a new DataFrame of listings in Brooklyn named `df_bk`
# and find how many listings in just Brooklyn of less than $100.
# First select condition.
c1 = df['neighbourhood_group'] == 'Brooklyn'
df_bk = df[c1]
c2 = df_bk['price'] < 100
len(df_bk[c2].id.value_counts())

# 3. Find how many listings there are in Brooklyn with a price less than 100.
c1 = df['neighbourhood_group'] == 'Brooklyn'
c2 = df['price'] < 100
len(df[c1&c2].id.value_counts())

# 4. Using `.isin()` select anyone that has the host name of Michael, David, John, and Daniel.

list_of_host_name = ['Michael', 'David', 'John', 'Daniel']
c1 = df.host_name.isin(list_of_host_name)
df[c1]

# 5. Create a new column called `adjusted_price` that has $100 added to every listing in Williamsburg.
# The prices for all other listings should be the same as the were before.
c1 = df['neighbourhood'] == 'Williamsburg'
df['adjusted_price'] = np.where(c1, df['price'] + 100, df['price'])
df[c1]

# 6. What % of the rooms are private, and what % of the rooms are shared.
c1 = df['room_type'] == 'Private room'
c2 = df['room_type'] == 'Shared room'
result = df[c1|c2].room_type.value_counts() / df.room_type.value_counts().sum()
result

# 1. Using `groupby`, count how many listings are in each neighbourhood_group.
df.groupby('neighbourhood_group')['neighbourhood_group'].count()

# 2. Using `groupby`, find the mean price for each of the neighbourhood_groups.
df.groupby('neighbourhood_group')['price'].mean()

# 3. Using `groupby` and `.agg()`, find the min and max price for each of the neighbourhood_groups.
df.groupby('neighbourhood_group')['price'].agg(['min', 'max'])

# 4. Using `groupby`, find the mean price for each room type in each neighbourhood_group.
groupby_cols = ['neighbourhood_group', 'room_type']
df.groupby(groupby_cols)['price'].mean()

# 5. Using `groupby` and `.agg()`, find the count, min, max, mean, median, and std of the prices
# for each room type in each neighbourhood_group.
groupby_cols = ['neighbourhood_group', 'room_type']
df.groupby(groupby_cols)['price'].agg(['count', 'min', 'max', 'mean', 'median', 'std'])

# 1. Load the `prices.csv` and the `n_listings.csv`
dfprices = pd.read_csv('data/prices.csv', sep=',')
dflistings = pd.read_csv('data/n_listings.csv', sep=';')

# 2. Do join that keeps all the records for each table.
dfjoined = pd.merge(dfprices, dflistings, on='neighbourhood_group', how='outer')
dfjoined
save_as = 'data/joined.csv'
dfjoined.to_csv(save_as, index=False)
dfjoined2 = pd.read_csv('data/joined.csv')
dfjoined2

#  1. Who was won Album of the Year in 2016?
dfgrammys = pd.read_csv('data/grammys.csv', sep=',')
dfgrammys.head()
c1 = dfgrammys['winner'] == True
c2 = dfgrammys['year'] == 2016
c3 = dfgrammys['category'] == 'Album of the Year'
dfgrammys[c1&c2&c3].workers

# 2. Who won Best Rap Album in 2009?
c1 = dfgrammys['winner'] == True
c2 = dfgrammys['year'] == 2009
c3 = dfgrammys['category'] == 'Best Rap Album'
dfgrammys[c1&c2&c3].workers

# 3. How many awards was Kendrick Lamar nomiated for, and how many did he win...?
# How many awards was Kendrick Lamar nomiated for
c1 = dfgrammys['workers'] == 'Kendrick Lamar'
dfnomiated = dfgrammys[c1].workers.value_counts()
dfnomiated

# 3. How many did he win?
c2 = dfgrammys['winner'] == True
dfwin = dfgrammys[c1&c2].workers.value_counts()
dfwin

