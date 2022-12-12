import pandas as pd
from ortools.sat.python import cp_model
import os
from PIL import Image, ImageOps, ImageStat
def sample_max_std_dev(df, column, sample_size):
    '''
    Returns sample_size rows from the given dataframe
    maximising the standard deviation of the specified column
    '''
    # Calculate length of dataframe
    n = len(df[column])
    # Set up or-tools
    model = cp_model.CpModel()

    # List of boolean variables for row selection
    row_selection = [model.NewBoolVar(f'{i}') for i in range(df.shape[0])]
    # Int variables for number of selected rows (length)
    # and the average and sum of these rows for the selected column
    # Greyscale pixels are 0-255
    average = model.NewIntVar(0, 255, 'average')                              
    total = model.NewIntVar(1, 255 * n, '')
    length = model.NewIntVar(1, n, '')

    # calculate length, sum, and average of selected column
    model.Add(length == pd.Series([1]*len(df)).dot(row_selection))
    model.Add(total == df[column].dot(row_selection))
    model.AddDivisionEquality(average, total, length)
    # The number of selected rows should equal sample size
    model.Add(length == sample_size)

    # loop over dataframe and calculate abs(x_i - u),
    # maximising this will maximise the std dev
    abs_diff = [model.NewIntVar(0, 255, '') for i in range(n)]
    for i in range(n):
        # zero series with only the current row set to 1
        row = pd.Series([0]*len(df))
        row[i] = 1
        # temporary interegers
        v = model.NewIntVar(-255, 255,'')
        x = model.NewIntVar(-255, 255, '')

        # calculate abs(x_i - u) but set to 0 if row not selected
        model.Add(v == df[column][i] - average)
        model.AddMultiplicationEquality(x, v, row.dot(row_selection))
        model.AddAbsEquality(abs_diff[i], x)
    
    # set objective
    model.Maximize(sum(abs_diff))

    # solve
    solver = cp_model.CpSolver()
    solver.Solve(model)

    # return the selected rows
    rows = [row for row in range(df.shape[0]) if solver.Value(row_selection[row])]
    return df.iloc[rows, :]

image_data = {'image_paths': [], 'mean_greyscale': []}

for root, dirs, files in os.walk("./images/"):
    for file in files:
        image_path = os.path.join(root, file)
        image_data['image_paths'].append(image_path)
        im = Image.open(image_path)
        im = ImageOps.grayscale(im)
        imstat = ImageStat.Stat(im)
        image_data['mean_greyscale'].append(round(imstat.mean[0]))

image_data = pd.DataFrame(image_data)
print(image_data)

result = sample_max_std_dev(image_data, 'mean_greyscale', 2)
print(result)