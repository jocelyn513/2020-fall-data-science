
# SQL:  Structured Query Language  Exercise

### Getting Started
1. Go to BigQuery UI https://console.cloud.google.com/bigquery
2. Add in the public data sets. 
	3. Click the Add Data icon
	4. Add any dataset
	5. `bigquery-public-data` should become visible and populate in the BigQuery UI. 
3. Add your queries where it says [YOUR QUERY HERE].
4. Make sure you add your query in between the triple tick marks. 
---
For this section of the exercise we will be using the `bigquery-public-data.austin_311.311_service_requests`  table. 

5. Write a query that tells us how many rows are in the table. 
	```
    SELECT
        COUNT(*)
    FROM
        `bigquery-public-data.austin_311.311_service_requests`
	```

6. Write a query that tells us how many _distinct_ values there are in the complaint_description column.
	``` 
    SELECT
        COUNT(DISTINCT complaint_description)
    FROM
        `bigquery-public-data.austin_311.311_service_requests`
	```
  
7. Write a query that counts how many times each owning_department appears in the table and orders them from highest to lowest.
	``` 
    SELECT
        DISTINCT owning_department, COUNT(unique_key) AS times_it_appeared
    FROM
        `bigquery-public-data.austin_311.311_service_requests`
    GROUP BY
        owning_department
    ORDER BY
        times_it_appeared DESC
	```

8. Write a query that lists the top 5 complaint_description that appear most and the amount of times they appear in this table. (hint... limit)
	```
    SELECT
        DISTINCT complaint_description, COUNT(unique_key) AS times_it_appeared
    FROM
        `bigquery-public-data.austin_311.311_service_requests`
    GROUP BY
        complaint_description
    ORDER BY
        times_it_appeared DESC
    LIMIT 5
	  ```
9. Write a query that lists and counts all the complaint_description, just for the where the owning_department is 'Animal Services Office'.
	```
    SELECT
        complaint_description, COUNT(unique_key) AS times_it_appeared
    FROM
        `bigquery-public-data.austin_311.311_service_requests`
    WHERE owning_department = 'Animal Services Office'
    GROUP BY complaint_description
    ORDER BY Times_it_appeared DESC
	```

10. Write a query to check if there are any duplicate values in the unique_key column (hint.. There are two ways to do this, one is to use a temporary table for the groupby, then filter for values that have more than one count, or, using just one table but including the  `having` function).
	```
    SELECT unique_key
    FROM `bigquery-public-data.austin_311.311_service_requests`
    GROUP BY unique_key
    HAVING COUNT(*) > 1
	```


### For the next question, use the `census_bureau_usa` tables.

1. Write a query that returns each zipcode and their population for 2000 and 2010. 
	```
    WITH TABLE_2000 AS (
    SELECT zipcode, SUM(population) AS population_2000 FROM `bigquery-public-data.census_bureau_usa.population_by_zip_2000` GROUP BY zipcode
    )
    , TABLE_2010 AS (
    SELECT zipcode, SUM(population) AS population_2010 FROM `bigquery-public-data.census_bureau_usa.population_by_zip_2010` GROUP BY zipcode
    )
    SELECT A.zipcode, population_2000, population_2010 FROM TABLE_2000 as A JOIN TABLE_2010 as B ON A.zipcode = B.zipcode
    ORDER BY A.zipcode
	```

### For the next section, use the  `bigquery-public-data.google_political_ads.advertiser_weekly_spend` table.
1. Using the `advertiser_weekly_spend` table, write a query that finds the advertiser_name that spent the most in usd. 
	```
    SELECT
    advertiser_name, SUM(spend_usd) AS total_spend_usd
    FROM
    `bigquery-public-data.google_political_ads.advertiser_weekly_spend`
    GROUP BY advertiser_name
    ORDER BY total_spend_usd DESC
    LIMIT 1
	```
2. Who was the 6th highest spender? (No need to insert query here, just type in the answer.)
	```
    TOM STEYER 2020
	```

3. What week_start_date had the highest spend? (No need to insert query here, just type in the answer.)
	```
    2020-09-13  21921700
	```

4. Using the `advertiser_weekly_spend` table, write a query that returns the sum of spend by week (using week_start_date) in usd for the month of August only. 
	```
    SELECT
    week_start_date, SUM(spend_usd) AS week_spend_usd
    FROM
    `bigquery-public-data.google_political_ads.advertiser_weekly_spend`
    WHERE EXTRACT(MONTH FROM week_start_date) = 8
    GROUP BY week_start_date
    ORDER BY week_start_date
	```
6.  How many ads did the 'TOM STEYER 2020' campaign run? (No need to insert query here, just type in the answer.)
	```
	50
	```
7. Write a query that has, in the US region only, the total spend in usd for each advertiser_name and how many ads they ran. (Hint, you're going to have to join tables for this one). 
	```
    WITH A AS (
    SELECT advertiser_name, SUM(spend_usd) AS total_spent_usd FROM `bigquery-public-data.google_political_ads.advertiser_weekly_spend`  GROUP BY advertiser_name
    )
    , B AS (
    SELECT advertiser_name, COUNT(*) AS num_of_ads FROM `bigquery-public-data.google_political_ads.advertiser_weekly_spend`  GROUP BY advertiser_name
    )
    SELECT A.advertiser_name, A.total_spent_usd, B.num_of_ads
    FROM A INNER JOIN B ON A.advertiser_name = B.advertiser_name
    WHERE A.total_spent_usd > 0
    ORDER BY A.total_spent_usd DESC
	```
8. For each advertiser_name, find the average spend per ad. 
	```
    SELECT
    advertiser_name, ROUND(AVG(spend_usd), 1) AS avg_spend
    FROM
    `bigquery-public-data.google_political_ads.advertiser_weekly_spend`
    GROUP BY advertiser_name
    ORDER BY avg_spend DESC
	```
10. Which advertiser_name had the lowest average spend per ad that was at least above 0. 
	``` 
	GREG CHANEY
    
    WITH A AS (
    SELECT
    advertiser_name, ROUND(AVG(spend_usd), 1) AS avg_spend
    FROM
    `bigquery-public-data.google_political_ads.advertiser_weekly_spend`
    GROUP BY advertiser_name
    ORDER BY avg_spend
    )
    SELECT
    advertiser_name, avg_spend
    FROM
    A
    WHERE avg_spend > 0
    ORDER BY avg_spend
	```
## For this next section, use the `new_york_citibike` datasets.

1. Who went on more bike trips, Males or Females?
	```
	Males
    
    SELECT gender, COUNT (*) AS num
    FROM `bigquery-public-data.new_york_citibike.citibike_trips`
    GROUP BY gender
	```
2. What was the average, shortest, and longest bike trip taken in minutes?
	```
    SELECT ROUND(AVG(tripduration/60),1) AS average, ROUND(MIN(tripduration/60),1) AS shortest, ROUND(MAX(tripduration/60),1) AS longest
    FROM `bigquery-public-data.new_york_citibike.citibike_trips`
	```

3. Write a query that, for every station_name, has the amount of trips that started there and the amount of trips that ended there. (Hint, use two temporary tables, one that counts the amount of starts, the other that counts the number of ends, and then join the two.) 
	```
    WITH A AS (
    SELECT start_station_name, COUNT (*) AS start_here
    FROM `bigquery-public-data.new_york_citibike.citibike_trips`
    GROUP BY start_station_name
    )
    , B AS (
    SELECT end_station_name, COUNT (*) AS end_here
    FROM `bigquery-public-data.new_york_citibike.citibike_trips`
    GROUP BY end_station_name
    )
    SELECT A.start_station_name AS station_name, A.start_here, B.end_here
    FROM A JOIN B ON A.start_station_name = B.end_station_name
    ORDER BY start_here DESC
	```
# The next section is the Google Colab section.  
1. Open up this [this Colab notebook](https://colab.research.google.com/drive/1kHdTtuHTPEaMH32GotVum41YVdeyzQ74?usp=sharing).
2. Save a copy of it in your drive. 
3. Rename your saved version with your initials. 
4. Click the 'Share' button on the top right.  
5. Change the permissions so anyone with link can view. 
6. Copy the link and paste it right below this line. 
	* YOUR LINK:  _____________https://colab.research.google.com/drive/10oJqWgObwiboWnqfZ8Qk9e0FzXuEGNrr?usp=sharing___________________
9. Complete the two questions in the colab notebook file. 
