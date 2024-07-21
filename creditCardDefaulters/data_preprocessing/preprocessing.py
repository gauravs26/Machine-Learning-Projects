import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import CategoricalImputer
from imblearn.over_sampling import RandomOverSampler
class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.
        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_unwanted_spaces(self,data):
        self.data = data

        try:
            self.df_without_spaces=self.data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)  # drop the labels specified in the columns
            return self.df_without_spaces
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in remove_unwanted_spaces method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            raise Exception()


    def remove_columns(self,data,columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception
        """
        self.data=data
        self.columns=columns
        try:
            self.useful_data=self.data.drop(labels=self.columns, axis=1) # drop the labels specified in the columns
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            raise Exception()

    def separate_label_feature(self, data, label_column_name):
        """
                        Method Name: separate_label_feature
                        Description: This method separates the features and a Label Coulmns.
                        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
                        On Failure: Raise Exception
                """
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            raise Exception()

    def is_null_present(self,data):
        """
                                Method Name: is_null_present
                                Description: This method checks whether there are null values present in the pandas Dataframe or not.
                                Output: Returns True if null values are present in the DataFrame, False if they are not present and
                                        returns the list of columns for which null values are present.
                                On Failure: Raise Exception
                        """
        self.null_present = False
        self.cols_with_missing_values=[]
        self.cols = data.columns
        try:
            self.null_counts=data.isna().sum() # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i]>0:
                    self.null_present=True
                    self.cols_with_missing_values.append(self.cols[i])
            if(self.null_present): # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv('preprocessing_data/null_values.csv') # storing the null column information to file
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(e))
            raise Exception()

    def impute_missing_values(self, data, cols_with_missing_values):
        """
                                        Method Name: impute_missing_values
                                        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
                                        Output: A Dataframe which has all the missing values imputed.
                                        On Failure: Raise Exception

                     """
        self.data= data
        self.cols_with_missing_values=cols_with_missing_values
        try:
            self.imputer = CategoricalImputer()
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            raise Exception()
    def scale_numerical_columns(self,data):
        """
                                                        Method Name: scale_numerical_columns
                                                        Description: This method scales the numerical values using the Standard scaler.
                                                        Output: A dataframe with scaled
                                                        On Failure: Raise Exception
                                     """
        self.data=data

        try:
            self.num_df = self.data.select_dtypes(include=['int64']).copy()
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns)
            return self.scaled_num_df

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            raise Exception()
    def encode_categorical_columns(self,data):
        """
                                                Method Name: encode_categorical_columns
                                                Description: This method encodes the categorical values to numeric values.
                                                Output: only the columns with categorical values converted to numerical values
                                                On Failure: Raise Exception
                             """

        try:
            self.cat_df = data.select_dtypes(include=['object']).copy()
            # Using the dummy encoding to encode the categorical columns to numericsl ones
            for col in self.cat_df.columns:
                self.cat_df = pd.get_dummies(self.cat_df, columns=[col], prefix=[col], drop_first=True)
            return self.cat_df

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            raise Exception()

    def handle_imbalanced_dataset(self,x,y):
        """
        Method Name: handle_imbalanced_dataset
        Description: This method handles the imbalanced dataset to make it a balanced one.
        Output: new balanced feature and target columns
        On Failure: Raise Exception
                                     """
        self.logger_object.log(self.file_object,
                               'Entered the handle_imbalanced_dataset method of the Preprocessor class')

        try:
            self.rdsmple = RandomOverSampler()
            self.x_sampled,self.y_sampled  = self.rdsmple.fit_sample(x,y)
            return self.x_sampled,self.y_sampled

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in handle_imbalanced_dataset method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            raise Exception()
