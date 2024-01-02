''' The code below will do the following.
1)Get the CSV data and make it to a Database.
2)Find the ideal functions using Least Squares method.
3)Mapping test points to best ideal function using the giving criterion.
4)Stores the new test data as per given formate to a table.
5)'Visualizing the comparison between Train and Best-Fit Columns.

'''

import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from SqlPackage import MakeConnection,IdealFunc
import matplotlib.pyplot as plt

IdealFuncPath=r"Dataset2\ideal.csv"
TrainPath=r"C:\Users\aliaz\IU\Dataset2\train.csv"
TestPath=r"C:\Users\aliaz\IU\Dataset2\test.csv"

ideal_dataset = MakeConnection(path=IdealFuncPath)
train_dataset = MakeConnection(path=TrainPath)
test_dataset = MakeConnection(path=TestPath)


test_data = test_dataset.data
test_data['ideal_function'] = ''

test_data['deviation'] = 0


ideal_dataset.CreatSql(file_name="Answer", title="ideal")
train_dataset.CreatSql(file_name="Answer", title="train")

ideal_positon = {
    'x': [], 'y1': [], 'y2': [], 'y3': [], 'y4': []}


'''Below Method Find Ideal Functions Based on Least Squares Method.

    Given a training dataset and an ideal dataset, this function determines the best-fit ideal function
    for each column (excluding the 'x' column) using the Least Squares method.

    Parameters:
        train_data (pd.DataFrame): The training dataset with 'x' and 'y' columns.
        ideal_set (pd.DataFrame): The ideal dataset with 'x' and 'y' columns.

    Returns:
        dict: A dictionary containing the column names as keys and the indices of the best-fit ideal functions
              (1-based index) as values.'''

def find_ideal_functions(train_data, ideal_set):
    ideal_position = {col: [] for col in ideal_set.columns}

    for train_col in train_data.columns:
        if train_col != 'x':
            # Calculate sum of squared deviations for each ideal column
            deviations = np.sum((ideal_set.loc[:, ideal_set.columns != 'x'].values - train_data[train_col].values[:, np.newaxis])**2, axis=0)

            # Find the index of the minimum deviation
            min_deviation_index = np.argmin(deviations)

            # Store the index (assuming indices are 0-based, add 1 for 1-based index)
            ideal_position[train_col] = min_deviation_index + 1

    return ideal_position

dict_pos= find_ideal_functions(train_dataset.data, ideal_dataset.data)
dict_pos.update({'x': 0})
dict = {key: value for key, value in dict_pos.items() if not isinstance(value, list) or value}
best_fit = ideal_dataset.data.iloc[:, list(dict.values())]

'''The Method below retrieves the corresponding 'y' value for a given 'x' from the ideal dataset.

    Parameters:
        x (float): The 'x' value for which the 'y' value is to be retrieved.
        ideal (pd.DataFrame): The ideal dataset containing 'x' and 'y' columns.

    Returns:
        float: The 'y' value corresponding to the given 'x'.
'''

def GetY(x, ideal):
    key = ideal['x'] == x
    try:
        return ideal.loc[key].iat[0, 1]
    except IndexError:
        raise IndexError
    

'''
    Map a test point to the best-fit ideal function based on given criteria.

    For a given test point (x, y), this function iterates through the best-fit ideal functions
    and selects the one with the minimum deviation that satisfies the given criteria.

    Parameters:
        test_point (list): A list containing the 'x' and 'y' values of the test point.
        best_fit (pd.DataFrame): The best-fit ideal functions dataset.
        train_dataset (MakeConnection): The training dataset.

    Returns:
        tuple: A tuple containing the name of the best-fit ideal function and its deviation from the test point.

    Raises:
        IndexError: If an index error occurs while retrieving values.

    
    '''


def mapping_points(test_point, ideal):
    no_of_ideal = None
    delta = float('inf')  # Initialize delta to positive infinity for comparison

    for ideal_set in best_fit.columns[1:5]:
        try:
            
            train_column = train_dataset.data.columns.intersection([ideal_set])
            deviation = IdealFunc(best_fit[ideal_set], train_dataset.data[train_column], ideal_set)
            largest_deviation = deviation.calc_max_deviation()
            y_location = GetY(test_point[0], best_fit[['x', ideal_set]])
            
        except IndexError:
            print(f"Index Error for {ideal_set}")
            raise IndexError

        current_deviation = abs(y_location - test_point[1])

        if current_deviation < largest_deviation * np.sqrt(2) and current_deviation < delta:
            no_of_ideal, delta = ideal_set, current_deviation

    return no_of_ideal, delta

           
for i, point in test_data.iterrows():
    ideal_column, yDel = mapping_points([point['x'], point['y']], best_fit)
    test_data.loc[i, 'ideal_function'] = ideal_column
    test_data.loc[i, 'deviation'] = yDel
    


engine = create_engine('sqlite:///Answer.db')
test_data.to_sql('test-data', con=engine, if_exists='replace', index=False)


new_column_names = {
    'x': 'x',
    'y42': 'y1',
    'y41': 'y2',
    'y11': 'y3',
    'y48': 'y4'
}
best_fit.rename(columns=new_column_names, inplace=True)


'''

    The function below iterates over each 'y' column in the dataframes and creates a separate plot
    for each column. It uses lines to represent the values of the columns from both dataframes and
    adds labels, legends, and titles for better interpretation.

    Parameters:
        df1 (pd.DataFrame): The first dataframe for comparison.
        df2 (pd.DataFrame): The second dataframe for comparison.

    Returns:
        None: Displays the plots using matplotlib.

    '''

def compare_individual_plots(df1, df2):
    # Iterate over each 'y' column
    for column in df1.columns[1:]:
        # Create a new figure for each 'y' column
        plt.figure()

        # Plot lines for df1
        plt.plot(df1['x'], df1[column], label=f'{column} from Best Fit dataframe', linestyle='-', marker='o')

        # Plot lines for df2
        plt.plot(df2['x'], df2[column], label=f'{column} from Trianing dataframe', linestyle='--', marker='x')

        # Adding labels and legend
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.title(f'Comparison of {column} Column from Best Fit and Train Set')

    # Display the plots
    plt.show()


compare_individual_plots(best_fit,train_dataset.data)


