import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/DisasterResponse.db')
df =  pd.read_sql_table('messages_table', engine)
X = df.message.values
y = df.iloc[:,5:]
