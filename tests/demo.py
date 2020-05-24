import pandas as pd
import timedisagg

expected_dataset = pd.read_csv("C:/Users/jstep/PycharmProjects/timedisagg/tests/sample_data.csv")

for conversion in ["sum", "average", "first", "last"]:
    td_obj = timedisagg.TempDisagg(conversion=conversion,method="dynamic-minrss")
    df_output = td_obj(expected_dataset)
    print(conversion)
    print(df_output)
    print(td_obj.rho_min)