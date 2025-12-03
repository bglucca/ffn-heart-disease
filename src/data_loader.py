from ucimlrepo import fetch_ucirepo
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from loguru import logger


class DataProcessor:
    def __init__(self, ucimlrepo_id: int = 45):
        logger.info(
            f"Fetching data from UCIML repository for ID: {ucimlrepo_id}"
        )
        self.raw_data = fetch_ucirepo(id=ucimlrepo_id)
        logger.info(
            f"Data fetched successfully. "
            f"Shape: {self.raw_data.data.features.shape}"
        )
        self.silver_data = self._process_data()

    def _list_variables_by_type(self, type_name: str):
        return self.raw_data.variables[
            self.raw_data.variables['type'] == type_name
        ]['name'].to_list()

    def _process_data(self):
        """
        Return silver level data. Data is preprocessed and ready
        for transformation. Dataframe is returned as pandas dataframe.
        """
        logger.info("Processing data...")
        # Concat datasets
        X, y = self.raw_data.data.features, self.raw_data.data.targets
        df = pd.concat([X, y], axis=1)

        # Impute missing values
        df.fillna(
            {'ca': df['ca'].mode()[0], 'thal': df['thal'].mode()[0]},
            inplace=True
        )
        df = df.astype({'ca': 'int64', 'thal': 'int64'})  # Remove decimal part

        # define categorical and integer variables
        categorical_vars = self._list_variables_by_type('Categorical')

        # Treat 'ca' as categorical
        if 'ca' not in categorical_vars:
            categorical_vars.append('ca')

        # Convert categorical variables to string
        df[categorical_vars] = df[categorical_vars].astype('str')

        return df

    def _transform_data(self):
        """Return gold level data. Data is transformed and ready
        for modeling. Data is returned as PyTorch Dataset.
        """
        logger.info("Transforming data for modeling...")
        # Get a copy of silver data to avoid modifying original
        df = self.silver_data.copy()

        # Get categorical and integer variable names
        categorical_vars = self._list_variables_by_type('Categorical')
        integer_vars = self._list_variables_by_type('Integer')

        # 'ca' should be treated as categorical
        # (already handled in _process_data)
        if 'ca' not in categorical_vars:
            categorical_vars.append('ca')
        if 'ca' in integer_vars:
            integer_vars.remove('ca')

        # Identify target variable (assuming 'num' for heart disease)
        target_col = 'num'

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        logger.info("Binarizing target variable...")
        y = (y > 0).astype(int)

        # Get categorical and integer columns in features
        cat_features = [col for col in categorical_vars if col in X.columns]
        int_features = [col for col in integer_vars if col in X.columns]

        # One-hot encode categorical variables
        X_cat_encoded = pd.get_dummies(
            X[cat_features], drop_first=True, dtype=int
        )

        # Scale integer variables
        scaler = StandardScaler()
        X_int_scaled = pd.DataFrame(
            scaler.fit_transform(X[int_features]),
            columns=int_features,
            index=X.index
        )

        # Combine processed features
        X_processed = pd.concat([X_int_scaled, X_cat_encoded], axis=1)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_processed.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        y_tensor = y_tensor[:, None]  # Convert to column vector

        # Create PyTorch Dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        logger.info(
            f"Data transformed successfully. "
            f"Samples: {len(dataset)}, Features: {X_tensor.shape[1]}"
        )
        return dataset

    def __call__(self):
        return self._transform_data()
