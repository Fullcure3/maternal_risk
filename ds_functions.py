import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, f_oneway, ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def missing_data_check(dataframe):
    """Summary of dtypes and null values to aid in data cleaning"""
    print(dataframe.info())
    print(dataframe.isnull().sum())


def summary_stats_barplot(dataframe, np_function, column, value):
    """Takes the dataframe, groups by column and sorts the data by value, according to the aggregate function passed in, for seaborn visualization\n
    Examples of functions are np.mean, np.median, etc"""
    
    # Summary stat to serve as a reference vertical line in the bar graph 
    summary_stat = np_function(dataframe[value])
    
    # Sort values by category in desending order to prep for creating an ordered barplot
    dataframe_sorted = (
                        dataframe.groupby(by=column, as_index=False)[value]
                        .agg(np_function)
                        .sort_values(by=value, ascending=False)
    )
    
    # Visualization of the choosen summary stat to identify trends 
    sns.barplot(data=dataframe_sorted, x=value, y=column)
    plt.axvline(summary_stat, linestyle='--', color='black')
    plt.show()  


def sorted_boxplot(dataframe, column, value):
    """Sorts the dataframe based on the category/column in descending order to create a sorted boxplot visualization"""
    dataframe.sort_values(by=column, ascending=False, inplace=True)
    sns.boxplot(data=dataframe, x=value, y=column)
    plt.show()


def unique_categories(dataframe, column):
    """Return a list of all unique categories for a given dataframe column"""
    categories = list(dataframe[column].unique())
    return categories


def column_std(dataframe, column, value):
    """Calculates the std of each category in a column to check for ANOVA Test assumption:\n
    The standard deviations of the groups should be equal\n
    Prints out the std of each category and prints out the ratio of the max std/min std"""
    
    #List of unique categories in a column for std calculations
    categories = unique_categories(dataframe, column)
    
    #Stores the std of each category to prep for ratio calculation
    standard_deviations = []

    #Prints out category and std for ANOVA Test std assumption 
    for category in categories:
        print(category, dataframe[dataframe[column] == category][value].std())
        standard_deviations.append(dataframe[dataframe[column] == category][value].std())
    
    ANOVA_std_ratio = max(standard_deviations) / min(standard_deviations)
    print(f"ANOVA std ratio for {value} is: {ANOVA_std_ratio}")


def zscore_normalization(dataframe, column, zscore_threshold=3):
    """Requires from scipy.stats import zscore\n
    Removes all rows that are considered outliers from a specific column based on a zscore threshold.\n
    The values above or below the threshold are considered outliers (default +/- 3)\n
    Prints out the number of rows removed.\n
    Returns the dataframe with outliers removed"""
    
    zscore_threshold
    # Abs to facilitate filtering of values that are above the zscore threshold
    dataframe_zscored = dataframe[(zscore(dataframe[column].abs()) < zscore_threshold)]

    records_removed = len(dataframe) - len(dataframe_zscored)
    print(f'{records_removed} rows removed')
    return dataframe_zscored


def anova_test(dataframe, column, value):
    """Preps data then performs an ANOVA Test based on the unique categories for the column of interest\n
    value = numeric data to test the choosen column categories against\n"""
    
    #List of unique categories from a column to prep for data filtering  
    categories = unique_categories(dataframe, column)

    #Data of each unique category to unpack as arguements for f_oneway (One way ANOVA Test)
    category_data = tuple([dataframe[value][dataframe[column] == category] for category in categories])
    
    #ANOVA Test to determine p-value significance
    fstat, pval = f_oneway(*category_data)
    print(pval)


def two_tail_ttest(dataframe, column, value, **kwargs):
    """Preps data then performs an two_tail TTest based on the binary categories for the column of interest\n
    value = numeric data to test the choosen column categories against\n"""
    
    #List of unique categories from a column to prep for data filtering  
    categories = unique_categories(dataframe, column)

    #Data of each unique category to unpack as arguements for ttest_ind (two tail ttest)
    category_data = tuple([dataframe[value][dataframe[column] == category] for category in categories])
    
    #Two Tail TTest to determine p-value significance
    fstat, pval = ttest_ind(*category_data, **kwargs)
    print(pval)


def tukeys_test(dataframe, column, value, pval_threshold=0.05):
    """Prints out the results of Tukey's Range Test to determine which pairings of an ANOVA Test are significant\n
    Uses standard significance threshold of 0.05 by default"""
       
    pval_threshold
    tukey_results = pairwise_tukeyhsd(dataframe[value], dataframe[column], pval_threshold)
    print(tukey_results)


def ln_transformation(dataframe, columns):
    """Performs a natural log transformation on the values from a specific column(s)\n
    Prints out the number of row removed to complete the transformation\n
    Returns a dataframe tranformed column(s)"""

    # Copy to perform ln transformation to preserve clean dataset
    dataframe_ln = dataframe.copy()

    #Remove all values <=0 to prep for transformation (log of value <= 0 is undefined)
    undefined = 0
    for column in columns:
        dataframe_ln = dataframe_ln[dataframe_ln[column] > undefined]

        # ln transformation for data profiling and ANOVA test
        dataframe_ln[column] = np.log(dataframe_ln[column])

    # Check number of records removed
    records_removed = len(dataframe) - len(dataframe_ln)
    print(f'{records_removed} records removed')

    return dataframe_ln


def unique_values(dataframe):
    """Examine values of all columns to find null or inappropriate values for data cleaning"""
    #Examine unique values for each column for data cleaning
    columns = dataframe.columns

    for column in columns:
        print(f"{column}: {dataframe[column].unique()}\n")


def corr_heatmap(dataframe, cmap='RdBu_r'):
    """Selects all numeric columns of a dataframe and displays a heatmap visualization"""
    # Filter by numeric columns to calculate pearson correlation
    dataframe_numeric = dataframe.select_dtypes(include=np.number)

    # Create a correlation matrix to visualize the correlations of the numeric columns
    corr_matrix = dataframe_numeric.corr()

    # Heatmap visualization to identify correlated variables
    sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, cmap=cmap)
    plt.show()


def corr_heatmap_by_category(dataframe, column, cmap='RdBu_r'):
    ## Using unique categories function create a filter for dataframes by category
    
    # Find all unique categories for a column to loop through
    categories = unique_categories(dataframe, column)

    for category in categories:
        # Filter by category and select only numeric columns to calculate pearson correlation
        filtered_dataframe = dataframe[dataframe[column] == category].select_dtypes(include=np.number)

        # Create a correlation matrix to visualize the correlations of the numeric columns
        corr_matrix = filtered_dataframe.corr()

        # Heatmap visualization to identify correlated variables
        sns.heatmap(corr_matrix, annot=True, vmin=-1, vmax=1, cmap=cmap)
        plt.title(category)
        plt.show()


def extract_features(dataframe, label):
    """Splits features and labels for future machine learning algorithms\n
    Label is the classification/regression column of interest\n
    Returns features, labels"""
    
    # Select features and outcomes for train_test_split
    features = dataframe.drop(columns=[label])
    labels = dataframe[label]

    # All features in the current dataset for verification 
    print(features.columns)

    return features, labels


def scale_features(ScalerModel, features, **kwargs):
    """Scales the features with the passed in scaler object\n
    Features are typically normalized to prevent one feature from having more weight than another, which can reduce model predictions\n
    Returns the scaled_features"""
    # Normalize the data to prevent one or more features from having more importance than other features during model creation
    scaler = ScalerModel(**kwargs)
    scaled_features = scaler.fit_transform(features)

    # Check for correct features 
    print(scaler.get_feature_names_out())

    return scaled_features


def sfs_selection_details(sfs, features):
    """Prints out chosen column indexes/names that scored the highest after sequential feature selection"""
    chosen_features_indexes = sfs.k_feature_idx_
    feature_column_names = features.iloc[:, list(chosen_features_indexes)].columns
    model_score = sfs.k_score_

    print(f"Column indexes chosen are: {chosen_features_indexes}")
    print(f"Column names chosen are: {feature_column_names}")
    print(f"Model score is: {model_score}")


def selected_feature_data(sfs, scaled_features):
    """Identifies highest scoring features for use in train_test_split and machine learning algorithms\n
    Returns the feature columns/data that scored the highest from the sfs"""
    
    # Convert numpy array of chosen feature indexes to list to filter scaled features dataframe
    chosen_features_indexes = list(sfs.k_feature_idx_)
    
    # Select all rows and specific columns using [rows, columns]
    features_sfs = scaled_features[:, chosen_features_indexes]
    
    return features_sfs


def sfs_feature_selection(MLModelClass, scaled_features, labels, **kwargs):
    """SFS to automate which features would improve the model's accuracy/effectiveness\n
    Returns the fitted sfs model, num_of_features"""
    sfs = SFS(estimator=MLModelClass(),
         **kwargs
         )

    sfs.fit(scaled_features, labels)
    num_of_features = len(sfs.k_feature_idx_)

    return sfs, num_of_features


def KNC_evaluation(X_train, X_test, y_train, y_test, **kwargs):
    """Creates a KNeighborsClaasifier and prints the scores of the training and testing sets to evaluate model performance\n
    Returns the fitted model to use for future predictions"""
    
    # Fit the model with train data and predict with test data to calculate accuracy of the model
    classifier = KNeighborsClassifier(**kwargs)
    classifier.fit(X_train, y_train)

    # Calculate the performance of the model to determine if overfitting/underfitting is occuring
    print(f"Accuracy score of training data: {classifier.score(X_train, y_train)}")
    print(f"Accuracy score of test data: {classifier.score(X_test, y_test)}")
    return classifier


def optimal_hyperparameters(ModelClass, tuned_parameters, X_train, y_train, **kwargs):
    """Iterates through all hyperparameter combinations passed in (tuned_parameters) to find the best performing model based on highest desired scoring method\n
    Returns a dictionary of best parameters for the highest scoring model"""
    
    # GridSearchCV to find the optimal n_neighbors based on desired scoring method 
    clf = GridSearchCV(estimator=ModelClass(),
                  param_grid=tuned_parameters,
                  **kwargs)
    
    clf.fit(X_train, y_train)
    best_parameters = clf.best_params_
    
    # Information on best performing model for evaluation
    print(clf.best_estimator_)
    print(f"Highest model {clf.scorer_} is: {clf.best_score_}")
    
    # Unpack dict (**best_parameters) to pass in arguements for the model
    return best_parameters