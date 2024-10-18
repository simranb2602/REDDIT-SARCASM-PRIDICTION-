
import tensorflow.keras.models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
 
class CustomLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=20):
        self.epochs = epochs
        self.model = self.create_model()
        self.input_layer = self.model.input
 
    def create_model(self):
        model = Sequential([
            LSTM(64, input_shape=(1, X_train_dense.shape[1])),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
 
    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, callbacks=[EarlyStopping(patience=3)])
 
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")
 
    def extract_features(self, X):
        feature_extractor = Model(inputs=self.model.input, outputs=self.model.layers[0].output)
        return feature_extractor.predict(X)
 
    def get_params(self, deep=True):
        return {'epochs': self.epochs}
 
    def set_params(self, **params):
        self.epochs = params['epochs']
        self.model = self.create_model()
        return self