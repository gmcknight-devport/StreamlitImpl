import pandas as pd
import streamlit as st
import requests
from plotly import graph_objs as go
from plotly.graph_objs import Line
import json

api_address = "https://mlspapi.azurewebsites.net"
model_names = ["LSTM", "SimpleRNN", "Bidirectional", "Conv1D", "GRU"]
activation_functions = ["tanh", "relu", "sigmoid", "softmax", "elu", "softsign", "softplus", "exponential"]

st.title("ML Stock Predictor (MLSP) Implementation")
warning_container = st.container()


def main():
    ml_accuracy, ml_predictions = ml_options()

    if ml_accuracy and ml_predictions:
        ml_results(ml_accuracy, ml_predictions)

    stat_predictions, model_summary = stats_options()

    if stat_predictions and model_summary:
        arima_results(stat_predictions, model_summary)

    # twit_data, sent_twit_results, fin_data, sent_fin_results = sentiment_options()
    fin_data, sent_fin_results = sentiment_options()

    # if sent_twit_results and sent_fin_results:
    #     sentiment_results(twit_data, sent_twit_results, fin_data, sent_fin_results)
    if sent_fin_results:
        fin_sentiment_results(fin_data, sent_fin_results)


# Return ticker values if valid
def ticker_validation(ticker, start_date, end_date):
    if ticker and start_date and end_date:
        ticker_list = {
            "ticker": ticker,
            "date_start": start_date.strftime("%Y-%m-%d"),
            "date_end": end_date.strftime("%Y-%m-%d")
        }

        ticker_final = {}
        for key, value in ticker_list.items():
            if value is not None:
                ticker_final[key] = value

        return ticker_final


# Set machine learning options expander
def ml_options():
    # ML Options
    expander_keras = st.expander("Set ML options")
    keras_form = expander_keras.form(key="keras")

    with keras_form:
        # Columns for stock selection
        ticker_cols = st.columns((1, 1, 1))
        ticker = ticker_cols[0].text_input("Ticker:")
        start_date = ticker_cols[1].date_input("Start date:")
        end_date = ticker_cols[2].date_input("End date:")

        st.header("Optional Parameters:")

        # First row
        model_cols1 = st.columns((1, 1, 1))
        model_name = model_cols1[0].selectbox(label="Select Model:", options=model_names, index=0)
        iterations = model_cols1[1].number_input("Iterations:", step=1)
        epochs = model_cols1[2].number_input("Epochs:", step=1)

        # Second Row
        model_cols2 = st.columns((1, 1, 1))
        num_inputs = model_cols2[0].number_input("Number of Inputs:", step=1)
        batch_size = model_cols2[1].number_input("Batch Size:", step=1)
        dropout = model_cols2[2].number_input("Dropout:")

        # Third Row
        model_cols3 = st.columns((1, 1, 1))
        optimiser = model_cols3[0].text_input("Optimiser:")
        loss = model_cols3[1].text_input("Loss:")
        activation = model_cols3[2].selectbox(label="Select Activation:", options=activation_functions, index=0)

        # Fourth Row
        model_cols4 = st.columns((1, 1, 1))
        ml_train_percentage = model_cols4[0].number_input("Train Percentage Split (0-1):")
        ml_time_step = model_cols4[1].number_input("Time Step:", step=1)
        ml_submitted = st.form_submit_button(label="Run")

        if ml_submitted:
            ticker_final = ticker_validation(ticker, start_date, end_date)
            if ticker_final:

                model_list = {
                    "iterations": iterations,
                    "epochs": epochs,
                    "num_inputs": num_inputs,
                    "batch_size": batch_size,
                    "dropout": dropout,
                    "optimiser": optimiser,
                    "loss": loss
                }

                model_final = {}
                for key, value in model_list.items():
                    if isinstance(value, int) and value != 0:
                        model_final[key] = value
                    elif isinstance(value, float) and value != 0.0:
                        model_final[key] = str(value)
                    elif isinstance(value, str) and value != '':
                        model_final[key] = str(value)

                data = {"ticker": ticker_final, "keras_model_options": model_final}

                keras_base_url = api_address + "/keras/?"
                request_url = keras_base_url

                url_vals = {"model_name": model_name, "activation": activation, "train_percentage":
                    ml_train_percentage, "time_step": ml_time_step}

                for key, val in url_vals.items():
                    if val and request_url != keras_base_url:
                        request_url += "&" + key + "=" + str(val)
                    else:
                        request_url += key + "=" + str(val)

                if request_url == api_address + "/keras/?":
                    request_url == api_address + "/keras/"

                data = json.dumps(data, indent=2)

                response = requests.post(request_url, data)
                results = response.json()

                return results[1], results[2]
            else:
                warning_container.warning("Enter a value for ticker, start date and end date")

    return {}, {}


# Set arima options expander
def stats_options():
    # Stats options
    expander_arima = st.expander("Set Stats options")
    arima_form = expander_arima.form(key="arima")

    with arima_form:
        # Columns for stock selection
        ticker_cols = st.columns((1, 1, 1))
        ticker = ticker_cols[0].text_input("Ticker:")
        start_date = ticker_cols[1].date_input("Start date:")
        end_date = ticker_cols[2].date_input("End date:")

        st.header("Optional Parameters:")

        # Sent options
        model_cols1 = st.columns((1, 1, 1))
        arima_train_percentage = model_cols1[0].number_input("Train Percentage Split (0-1):")

        arima_submitted = st.form_submit_button(label="Run")

        if arima_submitted:
            ticker_final = ticker_validation(ticker, start_date, end_date)
            if ticker_final:

                arima_base_url = api_address + "/statistical/arima"
                request_url = arima_base_url

                url_vals = {"train_percentage": arima_train_percentage}
                if url_vals.values():
                    request_url += "? train_percentage=" + str(arima_train_percentage)

                data = json.dumps(ticker_final, indent=2)

                response = requests.post(request_url, data)
                results = response.json()

                return results[0], results[1]
            else:
                warning_container.warning("Enter a value for ticker, start date and end date")

    return {}, {}


# Set sentiment analysis options expander
def sentiment_options():
    expander_sentiment = st.expander("Set Sentiment Analysis options")
    sentiment_form = expander_sentiment.form(key="sentiment")

    with sentiment_form:
        # Columns for stock selection
        ticker_cols = st.columns((1, 1, 1))
        ticker = ticker_cols[0].text_input("Ticker:")
        start_date = ticker_cols[1].date_input("Start date:")
        end_date = ticker_cols[2].date_input("End date:")

        sentiment_submitted = st.form_submit_button(label="Run")

        if sentiment_submitted:
            if ticker:
                sentiment_base_url = api_address + "/sentiment/fin-news"
                request_url = sentiment_base_url + "?ticker=" + ticker.upper()

                # url_vals = {"date_start": start_date, "date_end": end_date}
                #
                # for key, val in url_vals.items():
                #     if val:
                #         request_url += "&" + key + "=" + str(val)

                response = requests.get(request_url)
                print("Response is: ", response.text)
                results = response.json()

                error_message = {'detail': 'No tweets found'}

                if results != error_message:
                    # return results[0], results[1], results[2], results[3]
                    print("Results are: ", results)
                    return results[0], results[1]
                else:
                    warning_container.warning("Enter a start date before the end date")
            else:
                warning_container.warning("Enter a value for ticker, start date and end date")

    return {}, {}
    # return {}, {}, {}, {}


# Display keras results in container
def ml_results(accuracy: dict, predictions: dict):
    expander_ml_results = st.expander("Machine Learning Results")

    with expander_ml_results:
        # Display predictions
        results_cols = st.columns((1, 1))
        predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=["Prediction"])
        results_cols[0].dataframe(predictions_df)

        # Display accuracy
        accuracy_df = pd.DataFrame.from_dict(accuracy, orient='index', columns=["Accuracy"])
        df = accuracy_df.loc[["rmse", "mape", "min_max"]]
        results_cols[1].dataframe(df)

        # Plot graph)
        fig = go.Figure()
        fig.add_trace(Line(x=predictions_df.index, y=predictions_df['Prediction'], line_color='#000000',
                           name='Predictions Per Day'))
        fig.layout.update(title_text='ML Predictions', xaxis_rangeslider_visible=True, plot_bgcolor='#9fd8fb')

        st.plotly_chart(fig)


# Display arima results in container
def arima_results(predictions: dict, summary: dict):
    expander_arima_results = st.expander("Statistical Model (ARIMA) Results")

    with expander_arima_results:
        # Display predictions
        predictions_df = pd.DataFrame.from_dict(predictions, orient='index', columns=["Prediction"])
        st.dataframe(predictions_df)

        # Display accuracy
        st.text(summary)

        # Plot graph
        fig = go.Figure()
        fig.add_trace(Line(x=predictions_df.index, y=predictions_df['Prediction'], line_color='#000000',
                           name='Predictions Per Day'))
        fig.layout.update(title_text='ARIMA Predictions', xaxis_rangeslider_visible=True, plot_bgcolor='#9fd8fb')

        st.plotly_chart(fig)


def fin_sentiment_results(fin_data, fin_per_day):
    expander_sentiment_results = st.expander("Sentiment Analysis Results")

    with expander_sentiment_results:
        # Display data for each analysed piece of data for twitter and financial news
        data_cols = st.columns((1, 1))
        data_cols[0].header("Twitter")
        data_cols[0].text("No longer available due to changes from Mr. Musk.")

        data_cols[1].header("Financial News")
        fin_df = pd.DataFrame.from_dict(fin_data)
        fin_df['date'] = pd.to_datetime(fin_df['date'], yearfirst=True)  # , format='%Y-%m-%d')
        fin_df['date'] = fin_df['date'].dt.strftime('%Y/%m/%d')
        fin_table_df = data_cols[1].dataframe(fin_df, height=400)

        # Display data for compound sentiment per day
        compound_day_cols = st.columns((1, 1))
        # compound_day_cols[0].header("Twitter")

        compound_day_cols[1].header("Financial News")
        fin_day_df = pd.DataFrame.from_dict(fin_per_day)
        fin_day_df['date'] = pd.to_datetime(fin_day_df['date'], yearfirst=True)  # , format='%Y-%m-%d')
        fin_day_df['date'] = fin_day_df['date'].dt.strftime('%Y/%m/%d')
        fin_day_table_df = compound_day_cols[1].table(fin_day_df)

        # Plot data for compound sentiment results per day for titter and fin news on a single graph
        fig = go.Figure()
        fig.add_trace(Line(x=fin_day_df['date'], y=fin_day_df['compound'], line_color='#FFA500',
                           name='Financial news compound sentiment per day'))
        fig.layout.update(title_text='Sentiment Compound Scores', xaxis_rangeslider_visible=True,
                          plot_bgcolor='#9fd8fb')

        st.plotly_chart(fig)


# Display sentiment analysis results in container
def sentiment_results(twitter_data, twitter_per_day, fin_data, fin_per_day):
    expander_sentiment_results = st.expander("Sentiment Analysis Results")

    with expander_sentiment_results:
        # Display data for each analysed piece of data for twitter and financial news
        data_cols = st.columns((1, 1))
        data_cols[0].header("Twitter")
        twitter_df = pd.DataFrame.from_dict(twitter_data)
        twitter_table_df = data_cols[0].dataframe(twitter_df, height=400)

        data_cols[1].header("Financial News")
        fin_df = pd.DataFrame.from_dict(fin_data)
        fin_df['date'] = pd.to_datetime(fin_df['date'], yearfirst=True)  # , format='%Y-%m-%d')
        fin_df['date'] = fin_df['date'].dt.strftime('%Y/%m/%d')
        fin_table_df = data_cols[1].dataframe(fin_df, height=400)

        # Display data for compound sentiment per day
        compound_day_cols = st.columns((1, 1))
        compound_day_cols[0].header("Twitter")
        twitter_day_df = pd.DataFrame.from_dict(twitter_per_day)
        twitter_day_table_df = compound_day_cols[0].table(twitter_day_df)

        compound_day_cols[1].header("Financial News")
        fin_day_df = pd.DataFrame.from_dict(fin_per_day)
        fin_day_df['date'] = pd.to_datetime(fin_day_df['date'], yearfirst=True)  # , format='%Y-%m-%d')
        fin_day_df['date'] = fin_day_df['date'].dt.strftime('%Y/%m/%d')
        fin_day_table_df = compound_day_cols[1].table(fin_day_df)

        # Plot data for compound sentiment results per day for titter and fin news on a single graph
        fig = go.Figure()
        fig.add_trace(Line(x=twitter_day_df['date'], y=twitter_day_df['compound'], line_color='#C20808',
                           name='Twitter compound sentiment per day'))
        fig.add_trace(Line(x=fin_day_df['date'], y=fin_day_df['compound'], line_color='#FFA500',
                           name='Financial news compound sentiment per day'))
        fig.layout.update(title_text='Sentiment Compound Scores', xaxis_rangeslider_visible=True,
                          plot_bgcolor='#9fd8fb')

        st.plotly_chart(fig)


if __name__ == "__main__":
    main()
