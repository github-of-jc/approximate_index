# approximate_index
## How to run
This script takes in the the number of desired indices (int), the target index (str), and historical price data (.csv), and return a csv file with the symbols and weights that constitutes the replicated index. An additional feature is that the output csv will also have r value and number of times shuffled (see "how it works" for details) to indicate how good of a replication the script has created.

Run the script with the following command:

python3 approximate_index.py [n] [target_index] [historical_prices.csv] > [output.csv]

## How it works
The script does n steps:
1. convert the historical_prices.csv into dataframes
2. do the following for however many times you want:
2a. randomly picks n stocks and get a set of coefficients by doing nnls on the prices of these stocks
2b. store the coefficient and r value for this set of stocks
3. return the set of coefficient with the best r value
(4. optional: you can view the plot of the replicated index along with the target index)

## References
https://predictivemodeler.com/2020/05/09/non-negative-least-squares-regression/
