# Short_Rate_Prediction
predict 3m short rate that shifts out 6 months

Please see report.pdf

The ML model doesn't really work, but it was fun to work on.
The math model seems to be working

## To run this project
#### all *.csv and *.py files under the  `/math_model` folder need to be downloaded and saved in the same directory

change the starting month at line 203 in analyze.py 

change analyze window at line 218 in analyze.py

default values are `36` and `12` respectively, meaning starting at the 36th month, and predict next 6 months of short rate every 12 months

run `python analyze.py`

## Results

A prediction graph will appear along with the groundtruch interest rate curve

Some basic information of each prediction will be shown in terminal outputs

Overall RMSE(root mean square error) is shown at the end of the output

command such as `python analyze.py >> output.txt` could save the output if necessary

There are some outputs and graphs in this folder with names starting with `analyze_output*` and `traditional*`
