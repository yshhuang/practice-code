import pandas_datareader.data as web
import datetime
import pandas as pd

f = web.get_data_fred('SHA')
print(f)
