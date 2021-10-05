# Short_Rate_Prediction
predict 3m short rate that shifts out 6 months

Please see report.pdf

The ML model doesn't really work, but it was fun to work on.
The math model seems to be working

## To run the math project
#### all *.csv and *.py files under the  `/math_model` folder need to be downloaded and saved in the same directory

change the starting month at line 203 in analyze.py to preferred date

change analyze window at line 218 in analyze.py to preferred range

default values are `36` and `12` respectively, meaning starting at the 36th month, and predict next 6 months of short rate every 12 months

run `python analyze.py`




