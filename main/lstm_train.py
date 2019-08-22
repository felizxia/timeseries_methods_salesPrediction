import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from eval_metrics.eval import *

class LstmPipeline:
    def __init__(self):
        self.data = pd.read_csv('../preprocessed_data.csv')

    def split_sequence(self, sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return X, y

    def preprocessing(self, load=False):
        data = pd.read_csv('./preprocessed_data.csv')
        if not load:

            # choose a number of time steps
            # this time normalize data
            X_train, Y_train, X_test, Y_test = {}, {}, {}, {}
            n_steps = 7
            for s in data.Store.unique():
                X1, Y1, X2,  Y2 = [], [], [],[]
                for d in data.Dept.unique():
                    sample = data.loc[(data.Store == s) & (data.Dept == d)]
                    if len(sample) > n_steps:
                        sample_y = np.array(sample.y.values)
                    else:
                        continue
                    # sample_scale
                    scaler = StandardScaler()
                    sample_y_scale = scaler.fit_transform(sample_y.reshape(-1,1))
                    # for each store and each dept, we have the split sample res
                    # X, Y = split_sequence(sample_y, n_steps) # non scaled
                    x, y = self.split_sequence(sample_y_scale, n_steps)
                    X1 += x[:-1]
                    Y1 += y[:-1]
                    X2 += [x[-1]]
                    Y2 += [y[-1]]
                X_train[s] = np.array(X1)
                Y_train[s] = np.array(Y1)
                X_test[s] = np.array(X2)
                # X_test_original[s] = np.array(X2_)
                Y_test[s] = np.array(Y2)

            print('data preprocessing done..')
            np.save('processed_data/lstm/uni_lstm_X_train_norm.npy', X_train)
            np.save('processed_data/lstm/uni_lstm_Y_train_norm.npy', Y_train)
            np.save('processed_data/lstm/uni_lstm_X_test_norm.npy', X_test)
            np.save('processed_data/lstm/uni_lstm_Y_test_norm.npy', Y_test)
        else:
            X_train = np.load('processed_data/lstm/uni_lstm_X_train_norm.npy').item()
            Y_train = np.load('processed_data/lstm/uni_lstm_Y_train_norm.npy').item()
            X_test = np.load('processed_data/lstm/uni_lstm_X_test_norm.npy').item()
            Y_test = np.load('processed_data/lstm/uni_lstm_Y_test_norm.npy').item()

        return data, X_train, Y_train, X_test, Y_test

    def run(self):

        data, X_train, Y_train, X_test, Y_test = self.preprocessing(load=True)
        n_features = 1
        n_steps = 7
        res = defaultdict(dict)
        mape_accuracy = defaultdict(dict)
        mae_accuracy = defaultdict(dict)
        # train LSTM for each data/store
        for s in data.Store.unique():
            x_train, y_train, x_test,  y_test = X_train[s], Y_train[s], X_test[s],  Y_test[s]
            N = len(x_train)
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(x_train, y_train, epochs=20, verbose=0)
            yhat = model.predict(x_test, verbose=0) # prediction for each dept
            yhat = yhat.reshape(len(yhat),)
            depts = data.loc[(data.Store==s)]
            c = 0
            for d in depts.Dept.unique():
                if len(depts.loc[depts.Dept == d]) > n_steps:
                    mape_res = mape(yhat[c], y_test[c][0])
                    naive = x_test[c][-1]
                    mae = mae_scale(y_test[c][0], yhat[c], naive[0])
                    mape_accuracy[s][d] = mape_res
                    mae_accuracy[s][d] = mae
                    res[s][d] = yhat[c]
                    print('Store: '+ str(s) + ' Dept: ' + str(d) + ' mape: '+ str(mape_res) + ' mae_scaled_error: '+ str(mae))
                    c += 1
        np.save('np/norm/uni_lstm_res_norm.npy', res)
        np.save('np/norm/uni_lstm_mape_norm.npy', mape_accuracy)
        np.save('np/norm/uni_lstm_mae_norm.npy', mae_accuracy)

if __name__ == "__main__":
    Pipeline = LstmPipeline()
    Pipeline.run()
# only get 3 store whose dept prediction mae scaled error (<1)'s rate is > 50% of all data
# (2, 0.6052631578947368), (14, 0.5135135135135135), (23, 0.5733333333333334)

# after normalizetion: minmimal store has 40% of rate that are more accurate than naive prediction
