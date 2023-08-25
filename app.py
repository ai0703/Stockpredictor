import streamlit as st
import yfinance as yf
import ta
import numpy as np
from datetime import date, timedelta
from plotly import graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score
import warnings

# Filter out warnings
warnings.filterwarnings("ignore")

# Define start and end dates
TODAY = date.today()

# Set up Streamlit app title
st.title('Stock Prediction App')

# Choose stock ticker symbol
ticker = st.text_input('Enter stock ticker:', 'TSLA')

# Choose date range using slider
start_date = TODAY - timedelta(days=10*365)  # 10 years ago
end_date = TODAY

# Fetch stock data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate technical indicators
data['MA'] = ta.trend.SMAIndicator(data['Close'], window=10).sma_indicator()
data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()

# Drop NaN values
data.dropna(inplace=True)

# Split data into features (X) and target (y)
X = data.drop(['Close'], axis=1)
y = (data['Close'].shift(-1) < data['Close']).astype(int)  # Creating trend_label based on price change

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature Selection with SelectKBest
feature_selector = SelectKBest(score_func=f_classif, k=4)
X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
X_test_selected = feature_selector.transform(X_test_scaled)

# Naive Bayes Model
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train_selected, y_train)

# CNN Model
cnn_model = Sequential()
cnn_model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_selected.shape[1], 1)))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(1, activation='sigmoid'))

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape data for CNN input
X_train_cnn = X_train_selected.reshape(X_train_selected.shape[0], X_train_selected.shape[1], 1)
X_test_cnn = X_test_selected.reshape(X_test_selected.shape[0], X_test_selected.shape[1], 1)

num_epochs = 10
batch_size = 32

cnn_model.fit(X_train_cnn, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test_cnn, y_test))


# Evaluate CNN model
test_loss, test_accuracy = cnn_model.evaluate(X_test_cnn, y_test)

# Calculate F1-score for CNN model
y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype(int)
f1_cnn = f1_score(y_test, y_pred_cnn)

# Calculate F1-score for Naive Bayes model
y_pred_nb = naive_bayes_model.predict(X_test_selected)
f1_nb = f1_score(y_test, y_pred_nb)

# Display model accuracies and F1-scores
accuracy_nb = accuracy_score(y_test, naive_bayes_model.predict(X_test_selected))
st.write(f"Naive Bayes Classifier Accuracy: {accuracy_nb:.2f}")
st.write(f"Naive Bayes Classifier F1-Score: {f1_nb:.2f}")
st.write(f"CNN Model Accuracy: {test_accuracy:.2f}")
st.write(f"CNN Model F1-Score: {f1_cnn:.2f}")

# Plotting accuracy
accuracy_data = {
    'Model': ['Naive Bayes', 'CNN'],
    'Accuracy': [accuracy_nb, test_accuracy]
}

fig_accuracy = go.Figure(go.Bar(x=accuracy_data['Model'], y=accuracy_data['Accuracy']))
fig_accuracy.update_layout(title='Model Accuracy Comparison',
                           xaxis_title='Model',
                           yaxis_title='Accuracy',
                           template='plotly_white')
st.plotly_chart(fig_accuracy)

# ... (Rest of the code)


#Visualize stock data using Plotly
st.write("Stock Data and Accuracy History")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
fig.update_layout(title=f'{ticker} Stock Price Over Time',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  template='plotly_white')
st.plotly_chart(fig)

# Create an interactive section to predict future price movement
st.write("Predict Future Price Movement")

# Get user input for future prediction
future_data = data.iloc[-1].drop(['Close'])
future_data_input = st.text_input("Enter the future data values separated by commas:", ','.join(map(str, future_data)))

# Convert the input into an array and handle potential empty strings
future_data_values = [float(val.strip()) for val in future_data_input.split(',') if val.strip() != '']




# Select the prediction model
prediction_model = st.selectbox('Select Prediction Model', ['Naive Bayes', 'CNN', 'Ensemble'])

if st.button("Predict Price Movement"):
    future_data_scaled = scaler.transform(np.array(future_data_values).reshape(1, -1))
    future_data_selected = feature_selector.transform(future_data_scaled)

    if prediction_model == 'Naive Bayes':
        prediction_nb = naive_bayes_model.predict(future_data_selected)
        prediction_result = prediction_nb
    elif prediction_model == 'CNN':
        future_data_cnn = future_data_selected.reshape(1, -1, 1)
        prediction_cnn = cnn_model.predict(future_data_cnn)
        prediction_result = prediction_cnn
    elif prediction_model == 'Ensemble':
        future_data_cnn = future_data_selected.reshape(1, -1, 1)
        prediction_nb = naive_bayes_model.predict(future_data_selected)
        prediction_cnn = cnn_model.predict(future_data_cnn)
        ensemble_prediction = (prediction_nb + prediction_cnn) / 2
        prediction_result = ensemble_prediction

    if prediction_result[0] == 0:
        st.markdown('<span style="font-size:32px; color:red;">The model predicts that the stock price will go down. SELL!</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="font-size:32px; color:green;">The model predicts that the stock price will go up. BUY!</span>', unsafe_allow_html=True)
