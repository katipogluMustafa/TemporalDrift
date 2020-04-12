import pandas as pd


class Accuracy:

    @staticmethod
    def rmse(predictions) -> float:
        if type(predictions) is pd.DataFrame:
            number_of_predictions = 0
            sum_of_square_differences = 0.0
            for row in predictions.itertuples(index=False):
                # row[1] : actual rating, row[0] : prediction
                prediction = row[0]
                if prediction != 0:
                    sum_of_square_differences += (row[1] - prediction) ** 2
                    number_of_predictions += 1
            return sum_of_square_differences / number_of_predictions if number_of_predictions != 0 else 0
        elif type(predictions) is list:
            number_of_predictions = 0
            sum_of_square_differences = 0.0
            for prediction, actual in predictions:
                if prediction != 0:
                    sum_of_square_differences += (actual - prediction) ** 2
                    number_of_predictions += 1
            return sum_of_square_differences / number_of_predictions if number_of_predictions != 0 else 0



