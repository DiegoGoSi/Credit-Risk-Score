from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    # Take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the OrdinalEncoder and
    #     OneHotEncoder classes, then use the fitted models to transform all the
    #     datasets.

    Binary_Cols=[]
    Multi_Cols=[]
    input_df=[working_train_df,working_val_df,working_test_df]
    categorical_cols = working_train_df.select_dtypes(include='object').columns.tolist()
    for cat_col in categorical_cols:
        unique_values = working_train_df[cat_col].nunique()
        if unique_values>2:
            Multi_Cols.append(cat_col)
        else:
            Binary_Cols.append(cat_col)

   #OrdinalEncoder
    encoder = OrdinalEncoder()
    for df in input_df:
        for col in Binary_Cols:
            df[col] = encoder.fit_transform(df[[col]].fillna('Unknown'))

    #OneHotEncoder
    oh_encoder = OneHotEncoder(handle_unknown="ignore")
    oh_encoder.fit(working_train_df[Multi_Cols])
    train_enc_cols = oh_encoder.transform(working_train_df[Multi_Cols]).toarray()
    val_enc_cols = oh_encoder.transform(working_val_df[Multi_Cols]).toarray()
    test_enc_cols = oh_encoder.transform(working_test_df[Multi_Cols]).toarray()

    working_train_df.drop(columns=Multi_Cols, axis=1, inplace=True)
    working_val_df.drop(columns=Multi_Cols, axis=1, inplace=True)
    working_test_df.drop(columns=Multi_Cols, axis=1, inplace=True)

    working_train_df = np.concatenate([working_train_df.to_numpy(), train_enc_cols],axis=1)
    working_val_df = np.concatenate([working_val_df.to_numpy(), val_enc_cols],axis=1)
    working_test_df = np.concatenate([working_test_df.to_numpy(), test_enc_cols],axis=1)

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.

    imputer=SimpleImputer(missing_values=np.nan,strategy="median")
    imputer.fit(working_train_df)
    train=imputer.transform(working_train_df)
    val=imputer.transform(working_val_df)
    test=imputer.transform(working_test_df)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    scaler=MinMaxScaler(feature_range=(0,1))
    scaler.fit(train)
    train=scaler.transform(train)
    val=scaler.transform(val)
    test=scaler.transform(test)

    return train,val,test
