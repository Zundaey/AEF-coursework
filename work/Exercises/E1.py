#install in terminal:
# pip install tidyfinance


import pandas as pd
import numpy as np
import tidyfinance as tf
from plotnine import *


tf.download_data(
        domain="stock_prices", 
        symbols="AAPL",
        start_date="2000-01-01", 
          end_date="2023-12-31")

#0 Download data 
prices = tf.download_data(domain="stock_prices", symbols="AAPL",start_date="2000-01-03",end_date="2023-12-29")

prices.head().round(3)

#1 - Figur
prices.head(10).round(3)
apple_prices_figure = (
    ggplot(prices, aes(y="adjusted_close", x="date"))
    + geom_line()
    + labs(x="", y="", title="Apple stock prices from 2000 to 2023")
)
apple_prices_figure.show()

returns = (prices
    .sort_values("date") #sorter prices efter date
    .assign(ret=lambda x: x["adjusted_close"].pct_change()) #assign:Tilføj ny kolonne til data frame
    # x er "adju..close" og vi beregner pct. change (p_t-p_(t-1))/p_(t-1)
    .get(["symbol", "date", "ret"])# tilretter dataframe fra at være symbol,...,open,high osv til kun "symbol", "date", "ret"
)
returns


returns = (prices.sort_values("date").assign(ret))

