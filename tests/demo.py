import pandas as pd
from timedisagg.td import TempDisagg

expected_dataset = pd.read_csv("C:/Users/jstep/PycharmProjects/timedisagg/tests/sample_data.csv")

expected_dataset.head()

td_obj = TempDisagg(conversion="sum")
td_obj(expected_dataset)
