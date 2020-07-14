import numpy as np
import pandas as pd
from typing import Tuple, Dict#This will tell you what is the type of the things in the tuple
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

class AvocadoLSTM:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, n_neurons: int,
                 n_epochs: int, batch_size: int):
        self.train = train
        self.test = test
        self.n_neurons =n_neurons
        self.n_epochs = n_epochs
        self.batch_size = batch_size



    def create_feature_prediction_targets(self, X: pd.DataFrame, y: pd.DataFrame,
                                 feature_time_steps: int = 10) -> Tuple[np.array, np.array]:
        """
        Create prediction features and targets
        such that feature_time_steps rows from X
        are used to predict one row from y.

        Returns separate arrays for features and targets.
        """
        pass

    def prepare_dataset_for_LSTM(self, feature_time_steps: int = 10)-> Tuple[np.array, np.array, np.array, np.array]:
        """
        Calls self.create_feature_prediction_targets on train and test.
        Returns features and prediction targets for train and test, respectively,
        i.e. (X_train, y_train, X_test, y_test).
        Sets self.X_train, self.y_train etc to the results.
        """
        pass

    def inverse_scale(self, scaled_prediction: np.array, scaled_actual: np.array,
                      scaler: MinMaxScaler) -> pd.DataFrame:
        """
        Uses scaler.inverse_transform to transform scaled data back into its original units.
        Returns a DataFrame containing both prediction and actual data.
        """

    def evaluate_prediction(self, prediction_actual: pd.DataFrame) -> Dict[str, float]:
        """
        Uses the actual data in prediction_actual to evaluate its prediction,
        Returns a dictionary with keys giben by error matrics (Currently 'mae', 'rmse')
        """
        pass

    def build_LSTM(self):
        """
        Deferred to StatefulAvocadoLSTM and StatelessAvocadoLSTM"""
        raise NotImplementedError("Deferred to StatefulAvocadoLSTM and StatelessAvocadoLSTM")


class StatefulAvocadoLSTM(AvocadoLSTM):

    def build_LSTM(self, n_neurons: int, batch_size: int, dropout_ratio: float):
        '''
        Builds a sequential neural network consisting of
        * a stateful LSTM layer with n_neurons, given batch size and train data shape
        * a dropout layer using dropout_ratio
        * a dense layer with a single node (predict 1 step ahead)
        et self.model to the result.
        '''
        model = Sequential()
        model.add(LSTM(self.n_neurons,
                       batch_input_shape= (batch_size, self.X_train.shape[1], self.X_train.shape[2]),
                       stateful=True))
        model.add(Dropout(dropout_ratio))
        model.add((Dense(1)))
        model.compile(loss='mse', optimizer='adam')
        self.model = model
        pass



    def fit_LSTM(self, verbose: bool = False):
        '''
        Fits self.model to the train data.
        '''
        for epoch in range(self.n_epochs):
            self.model.fit(self.X_train, self.y_train, epochs =1,
                           batch_size = self.batch_size, shuffle = False,
                          verbose = verbose)
            self.model.reset_states()
        pass

    def save_model(self, filepath: str = 'Stateful_LSTM_model.h5'):
        '''
        Saves the fitted model to h5 at filepath.
        '''
        self.model.save(filepath = self.filepath)



    def prepare_batch_for_prediction(self, input_arr: np.array,
                                    n_input: int, n_features: int) -> np.array:
        '''
        Prepare data format for predicting by
        creating batch from input_arr which is a 1D array
        '''
        batch = []
        for i in range(n_input):
            lookback = input_list[i: i + n_input].reshape((n_input, n_features))
            batch.append(lookback)
        return np.array(batch)



    def predict_from_history(self):
        '''
        In order to predict each point in the test data,
        feed the ten previous data points into self.model.predict.
        Returns an array containing all test predictions except the first ten.
        '''
        n_input = self.X_train.shape[1]
        n_features = self.X_train.shape[2]
        flat_list = self.train.values[-19:]
        stateful_prediction_list = []
        for index_to_predict  in range(len(self.test)):
            batch = self.prepare_batch_for_prediction(flat_list, n_input, n_features)
            prediction = self.model.predict(batch)
            stateful_prediction_list.append(predictions[-1][0])
            flat_list = flat_list[1:]
            flat_list = np.append(flat_list, self.test.values[index_to_predict])
        return np.array(stateful_prediction_list)


    def run(self):
        self.prepare_dataset_for_LSTM()
        self.build_LSTM()
