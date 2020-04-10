import pandas as pd


class Accuracy:

    @staticmethod
    def rmse(predictions) -> float:
        if type(predictions) is pd.DataFrame:
            number_of_predictions = predictions.count()
            sum_of_square_differences = 0.0
            for row in predictions.itertuples(index=False):
                prediction_rating = row[0]   # row[0] : prediction, row[1] : actual rating
                if prediction_rating != 0:
                    sum_of_square_differences += (row[1] - prediction_rating) ** 2
            result = sum_of_square_differences / number_of_predictions
            return result[0]
        elif type(predictions) is list:
            number_of_predictions = len(predictions)
            sum_of_square_differences = 0.0
            for prediction, actual in predictions:
                sum_of_square_differences += (actual - prediction) ** 2
            result = sum_of_square_differences / number_of_predictions
            return result


