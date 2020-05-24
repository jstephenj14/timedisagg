# Temporal Disaggregation of Time Series Data

This package implements time disaggregation models inspired heavily by [R's tempdisagg package](https://github.com/christophsax/tempdisagg). The package
currently has the following methods implemented:

- Chow-Lin (max. log)
- Chow-Lin (min. RSS)
- Litterman (max. log)
- Litterman (min. RSS)
- Chow-Lin (dynamic max. log)
- Chow-Lin (dynamic min. RSS)

Details on these models as well as the underlying math are available at [this article](https://journal.r-project.org/archive/2013/RJ-2013-028/RJ-2013-028.pdf) written by the R package's authors.

These models ensure that the sum, the average, the first or the last value of the resulting high frequency series is consistent with the low frequency series.
The conversion type can also be adjusted by the function call.

### Input dataset format

The traditional example of disaggregating `sales.a` against `exports.q` available under the R package's vignette is discussed here.

The input dataset to the TempDisagg's object call should have the following format:

```
     index  grain            X            y
0     1972      1   1432.63900          NaN
1     1972      2   1456.89100          NaN
2     1972      3   1342.56200          NaN
3     1972      4   1539.39400          NaN
4     1973      1   1535.75400          NaN
5     1973      2   1578.45800          NaN
6     1973      3   1574.72400          NaN
7     1973      4   1652.17100          NaN
8     1974      1   2047.83400          NaN
9     1974      2   2117.97100          NaN
10    1974      3   1925.92600          NaN
11    1974      4   1798.19000          NaN
12    1975      1   1818.81700   136.702329
13    1975      2   1808.22500   136.702329
14    1975      3   1649.20600   136.702329
15    1975      4   1799.66500   136.702329
16    1976      1   1985.75300   151.056074
17    1976      2   2064.66300   151.056074
18    1976      3   1856.38700   151.056074
19    1976      4   1919.08700   151.056074
..     ...    ...          ...          ...
152   2010      1  19915.79514   988.309676
153   2010      2  19482.48000   988.309676
154   2010      3  18484.64900   988.309676
155   2010      4  18026.46869   988.309676
156   2011      1  19687.52100          NaN
157   2011      2  18913.06608          NaN
```

The `index` column holds the low-frequency time periods from which high-frequency time series (`grain`) are generated. 
`y` is the data to be disaggregated and `X` is the high-frequency regressor. In this case, `X` is the quarter-level 
export numbers while `y` holds yearly sales numbers. This data is available under `tests/sample_data.csv` along with a `demo-run.py`.

Note that the current package does not accept `Nan` values in any of these columns except for valid backcasting and forecasting periods in the `y` column.
For example, in the above case, the package is expected to provide backcasted values for the 12 quarters from 1972 to 1974 and 2 forecasted quarters for 2011.

### Walkthrough

The file for this walkthrough is available as `tests/demo_run.py`. Ensure to `pip install timedisagg` before running.
 
```

import pandas as pd
from timedisagg.td import TempDisagg

expected_dataset = pd.read_csv("./tests/sample_data.csv")

td_obj = TempDisagg(conversion="sum", method="chow-lin-maxlog")
final_disaggregated_output = td_obj(expected_dataset)

print(final_disaggregated_output.head()

Output:
   index  grain         X   y      y_hat
0   1972      1  1432.639 NaN  21.656879
1   1972      2  1456.891 NaN  22.219737
2   1972      3  1342.562 NaN  20.855413
3   1972      4  1539.394 NaN  23.937916
4   1973      1  1535.754 NaN  24.229008
```
`y_hat` holds the disaggregated values of `y`.

Detailed documentation will follow, but for now the different methods can be referred by the following tags:

```
"chow-lin-maxlog" 
"chow-lin-minrss-ecotrim" 
"chow-lin-minrss-quilis" 
"litterman-maxlog" 
"litterman-minrss" 
"dynamic-maxlog" 
"dynamic-minrss"
```
And conversions can be selected by `sum`,`average`,`first` and `last` as available in the R package.

