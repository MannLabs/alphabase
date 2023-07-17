import pandas as pd
import numpy as np

from typing import Union, List

class Column():
    def __init__(
            self, 
            name : str, 
            type : Union[str, type, np.dtype], 
            allow_NaN : bool = False, 
            allow_inf : bool = False
        ):
        """
        Base class for validating a single column.
        The column is safely cast to the specified type inplace.
        NaN and inf values are checked.

        Parameters
        ----------

        name: str
            Name of the column

        type: Union[str, type, np.dtype]
            Type of the column

        allow_NaN: bool
            If True, allow NaN values

        allow_inf: bool
            If True, allow inf values

        Properties
        ----------

        name: str
            Name of the column

        type: Union[type, np.dtype]
            Type of the column
        
        """

        self.name = name

        if isinstance(type, str):
            self.type = np.dtype(type)
        else:
            self.type = type

        self.allow_NaN = allow_NaN
        self.allow_inf = allow_inf

    def __call__(
            self, 
            df : pd.DataFrame
            ):
        """
        Validates the column.

        Parameters
        ----------

        df: pd.DataFrame
            Dataframe which contains the column.

        """
        if df[self.name].dtype != self.type:
            if np.can_cast(df[self.name].dtype, self.type):
                df[self.name] = df[self.name].astype(self.type)
            else:
                raise ValueError(f"Validation failed: Column {self.name} of type {_get_type_name(df[self.name].dtype)} cannot be cast to {_get_type_name(self.type)}")
            
        if not self.allow_NaN:
            if df[self.name].isna().any():
                raise ValueError(f"Validation failed: Column {self.name} contains NaN values")
            
        if not self.allow_inf:
            if not np.isfinite(df[self.name]).all():
                raise ValueError(f"Validation failed: Column {self.name} contains inf values")

class Optional(Column):
    """
    Optional column to be validated.
    If the column is not present in the dataframe, the validation is skipped.
    """
    def __init__(self, *args, **kwargs):
        """
        Optional column
        
        Parameters
        ----------
        
        name: str
            Name of the column
                
        type: type
            Type of the column
            
        """

        super().__init__( *args, **kwargs)
        

    def __call__(
            self, 
            df : pd.DataFrame
        ):
        """
        Casts the column to the specified type if it is present in the dataframe

        Parameters
        ----------

        df: pd.DataFrame
            Dataframe to validate

        """

        if self.name in df.columns:
            super().__call__(df)

class Required(Column):
    """
    Required column to be validated.
    If the column is not present in the dataframe, the validation fails.
    """
    def __init__(self, *args, **kwargs):
        """
        Required column

        Parameters
        ----------

        name: str
            Name of the column

        type: type
            Type of the column

        """
        super().__init__(*args, **kwargs)

    def __call__(
            self, 
            df : pd.DataFrame
        ):
        """
        Casts the column to the specified type if it is present in the dataframe

        Parameters
        ----------

        df: pd.DataFrame
            Dataframe to validate

        """

        if self.name in df.columns:       
            super().__call__(df)
        else:
            raise ValueError(f"Validation failed: Column {self.name} is not present in the dataframe")

class Schema():
    def __init__(
            self, 
            name : str, 
            properties: List[Column]):
        """
        Schema for validating dataframes

        Parameters
        ----------

        name: str
            Name of the schema

        properties: list
            List of Property objects

        """

        self.name = name
        self.schema = properties
        for column in self.schema:
            if not isinstance(column, Column):
                raise ValueError(f"Schema must contain only Property objects")

    def __call__(self, df):
        """
        Validates the dataframe

        Parameters
        ----------

        df: pd.DataFrame
            Dataframe to validate

        """

        for column in self.schema:
            column(df)

def _get_type_name(
        type : Union[str, type, np.dtype]) -> str:
    """
    Returns the human readable name of the type

    Parameters
    ----------

    type: Union[str, type, np.dtype]
        Type to get the name of

    Returns
    -------

    name: str
        Human readable name of the type

    """
    if isinstance(type, str):
        return type
    elif isinstance(type, np.dtype):
        return type.name
    else:
        return type.__name__