import math
import numpy as np
import pandas as pd

class Accuracy:
  """
  Accuracy class provides deffirent metrics in order to measure accuracy of our analysis.

  Supported Measures:
  rmse, accuracy, balanced accuracy, informedness, markedness,
  f1, mcc, precision, recall, specificity, NPV and
  other threshold measures where we round ratings less than 3.5 to min rating, upper to max rating and use supported
  measures on this data.
  """

  @staticmethod
  def rmse(predictions) -> float:
    """
    Calculate Root Mean Square Error of given list or Dataframe of (prediction, actual) data.

    In case rmse value is found 0, it is returned as 0.001 to differentiate between successfull rmse
    calculation and erroneous calculations where no prediction data is provided.
    """

    # In case dataframe of predictions wher each row[0]=prediction, row[1]=actual rating
    if type(predictions) is pd.DataFrame:
      number_of_predictions = 0
      sum_of_square_differences = 0.0
      for row in predictions.itertuples(index=False):
        prediction = row[0]
        # In case valid prediction is made(0 is invalid, minimum 0.5 in movielens dataset)
        if prediction != 0:
          # Round the ratings to the closest half or exact number
          # since movielens dataset only containst ratings 0.5, 1, 1.5,..., 4, 4.5, 5
          actual = Accuracy.half_round_rating(row[1])
          prediction = Accuracy.half_round_rating(prediction)

          sum_of_square_differences += (actual - prediction) ** 2
          number_of_predictions += 1

        if number_of_predictions == 0:
          return 0
        rmse_value = sum_of_square_differences / number_of_predictions
      return rmse_value if rmse_value != 0 else 0.001
    # In case list of predictions where each element is (prediction, actual)
    elif type(predictions) is list:
      number_of_predictions = 0
      sum_of_square_differences = 0.0
      for prediction, actual in predictions:
        if prediction != 0:  # if the prediction is valid
          actual = Accuracy.half_round_rating(actual)
          prediction = Accuracy.half_round_rating(prediction)

          sum_of_square_differences += (actual - prediction) ** 2
          number_of_predictions += 1

      if number_of_predictions == 0:
        return 0

      rmse_value = sum_of_square_differences / number_of_predictions
      return rmse_value if rmse_value != 0 else 0.001
    return 0

  @staticmethod
  def threshold_accuracy(predictions) -> float:
    """
    Threshold accuracy is the rate of sucessful prediction when we round
    ratings between 0.5 and 3.5 to the lowest rating(0.5) ,
    ratings between 3.5 and 5 to the highest rating(5)

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    """

    if type(predictions) is pd.DataFrame:
      number_of_predictions = 0
      number_of_hit = 0
      for row in predictions.itertuples(index=False):
        # row[1] : actual rating, row[0] : prediction
        prediction = row[0]
        if prediction != 0:
          actual = Accuracy.threshold_round_rating(row[1])
          prediction = Accuracy.threshold_round_rating(prediction)

          if actual == prediction:
            number_of_hit += 1
          number_of_predictions += 1
      return number_of_hit / number_of_predictions if number_of_predictions != 0 else 0
    elif type(predictions) is list:
      number_of_predictions = 0
      number_of_hit = 0
      for prediction, actual in predictions:
        if prediction != 0:
          actual = Accuracy.threshold_round_rating(actual)
          prediction = Accuracy.threshold_round_rating(prediction)

          if actual == prediction:
            number_of_hit += 1

          number_of_predictions += 1
      return number_of_hit / number_of_predictions if number_of_predictions != 0 else 0
    return 0

  @staticmethod
  def threshold_analize(predictions):
    """
    Analize the threshold predictions with all metrics found in the Accuracy class.
    """

    TP, FN, FP, TN = Accuracy.threshold_confusion_matrix(predictions)
    precision = Accuracy.precision(TP, FP)  # also called PPV
    recall = Accuracy.recall(TP, FN)  # also called TPR
    specificity = Accuracy.specificity(FP, TN)  # also called TNR
    NPV = Accuracy.negative_predictive_value(FN, TN)

    accuracy = Accuracy.accuracy(TP, FN, FP, TN)
    balanced_accuracy = Accuracy.balanced_accuracy(TPR=recall, TNR=specificity)
    informedness = Accuracy.informedness(TPR=recall, TNR=specificity)
    markedness = Accuracy.markedness(PPV=precision, NPV=NPV)

    f1 = Accuracy.f_measure(precision, recall)
    mcc = Accuracy.mcc(TP, FN, FP, TN)

    output = {
      "accuracy"         : round(accuracy, 3),
      "balanced_accuracy": round(balanced_accuracy, 3),
      "informedness"     : round(informedness, 3),
      "markedness"       : round(markedness, 3),
      "f1"               : round(f1, 3),
      "mcc"              : round(mcc, 3),
      "precision"        : round(precision, 3),
      "recall"           : round(recall, 3),
      "specificity"      : round(specificity, 3),
      "NPV"              : round(NPV, 3)
    }

    return output

  @staticmethod
  def analize(predictions):
    """
    Analize the threshold predictions with all metrics found in the Accuracy class.

    https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

    Returns analysis for each class as list
    :return: accuracy, balanced_accuracy, informedness, markedness, f1, mcc, precision, recall, specificity, NPV
    """
    confusion_mtr = Accuracy.confusion_matrix(predictions)

    # Use macro averaging (https://stats.stackexchange.com/questions/187768/matthews-correlation-coefficient-with
    # -multi-class)
    precision = [0] * 10  # 10 is the number of classes found
    recall = [0] * 10  # 0.5 -> Class 0 , 1 -> Class 1, 1.5 -> Class 2 ....
    specificity = [0] * 10
    NPV = [0] * 10

    accuracy = [0] * 10
    balanced_accuracy = [0] * 10
    informedness = [0] * 10
    markedness = [0] * 10

    f1 = [0] * 10
    mcc = [0] * 10

    for i in range(0, 10):  # For Each Class
      TP, FN, FP, TN = Accuracy.confusion_matrix_one_against_all(confusion_mtr, i)
      precision[i] = Accuracy.precision(TP, FP)  # also called PPV
      recall[i] = Accuracy.recall(TP, FN)  # also called TPR
      specificity[i] = Accuracy.specificity(FP, TN)  # also called TNR
      NPV[i] = Accuracy.negative_predictive_value(FN, TN)

      accuracy[i] = Accuracy.accuracy(TP, FN, FP, TN)
      balanced_accuracy[i] = Accuracy.balanced_accuracy(TPR=recall[i], TNR=specificity[i])
      informedness[i] = Accuracy.informedness(TPR=recall[i], TNR=specificity[i])
      markedness[i] = Accuracy.markedness(PPV=precision[i], NPV=NPV[i])

      f1[i] = Accuracy.f_measure(precision[i], recall[i])
      mcc[i] = Accuracy.mcc(TP, FN, FP, TN)

    output = {
      "accuracy"         : Accuracy.round_list_elements(accuracy, 3),
      "balanced_accuracy": Accuracy.round_list_elements(balanced_accuracy, 3),
      "informedness"     : Accuracy.round_list_elements(informedness, 3),
      "markedness"       : Accuracy.round_list_elements(markedness, 3),
      "f1"               : Accuracy.round_list_elements(f1, 3),
      "mcc"              : Accuracy.round_list_elements(mcc, 3),
      "precision"        : Accuracy.round_list_elements(precision, 3),
      "recall"           : Accuracy.round_list_elements(recall, 3),
      "specificity"      : Accuracy.round_list_elements(specificity, 3),
      "NPV"              : Accuracy.round_list_elements(NPV, 3)
    }

    return output

  @staticmethod
  def round_list_elements(l, precision):
    """
    :param l: list of floats
    :param precision: precision after dot
    """
    return [round(x, precision) for x in l]

  @staticmethod
  def accuracy_multi_class(confusion_mtr):
    length = len(confusion_mtr)
    numenator = 0
    denuminator = 0
    for i in range(length):
      for j in range(length):
        temp = confusion_mtr[i][j]
        denuminator += temp
        if i == j:
          numenator += temp
    return numenator / denuminator

  @staticmethod
  def accuracy(TP, FN, FP, TN):
    return (TP + TN) / (TP + FN + FP + TN)

  @staticmethod
  def balanced_accuracy(TPR, TNR):
    """
    :param TPR : True Positive Rate or recall or sensitivity
    :param TNR : True Negative Rate or specificity or  selectivity
    """
    return (TPR + TNR) / 2

  @staticmethod
  def informedness(TPR, TNR):
    """
    :param TPR : True Positive Rate or recall or sensitivity
    :param TNR : True Negative Rate or specificity or  selectivity
    """
    return TPR + TNR - 1

  @staticmethod
  def markedness(PPV, NPV):
    """
    :param PPV: Positive Predictive Value also known as precision
    :param NPV: Negative Predictive Value
    """
    return PPV + NPV - 1

  @staticmethod
  def precision(TP, FP):
    """
    Also called as precision or positive predictive value (PPV)

    Precision = TP / (TP + FP) for binary class
    Precision = TP / (All Predicted Positive) for multi class
    """
    denuminator = TP + FP
    return TP / denuminator if denuminator != 0 else 0

  @staticmethod
  def negative_predictive_value(FN, TN):
    """
    NPV = TN / (TN + FN) for binary class
    NPV = TN / (All Predicted Negative) for multi class
    """
    denuminator = TN + FN
    return TN / denuminator if denuminator != 0 else 0

  @staticmethod
  def recall(TP, FN):
    """
    Also called as sensitivity, recall, hitrate, or true positive rate(TPR)
    Recall = TP / (TP + FN) for binary class
    Recall = TP / (All Actual Positive) for multi class
    """
    denuminator = TP + FN
    return TP / denuminator if denuminator != 0 else 0

  @staticmethod
  def specificity(FP, TN):
    """
    Also called as specificity, selectivity or true negative rate (TNR)
    specificity = TN / (TP + FN) for binary class
    specificity = TN / (All Actual Negative) for multi class
    """
    denuminator = FP + TN
    return TN / denuminator if denuminator != 0 else 0

  @staticmethod
  def f_measure(precision, recall):
    """
    F-Measure is the harmonic mean of the precision and recall.
    """
    sum_of_both = precision + recall
    return (2 * precision * recall) / sum_of_both if sum_of_both != 0 else 0

  @staticmethod
  def mcc(TP, FN, FP, TN):
    """
    MCC(Matthews Correlation Coefficient)
    """
    # Calulate Matthews Correlation Coefficient
    numenator = (TP * TN) - (FP * FN)
    denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    denominator = math.sqrt(denominator) if denominator > 0 else 0
    return numenator / denominator if denominator != 0 else 0

  @staticmethod
  def confusion_matrix_one_against_all(confusion_mtr, class_i):
    """
    Create binary confusion matrix out of multi-class confusion matrix

    Positive Class: class_i
    Negative Class: non class_i

    TP: True Positive   FN: False Negative
    FP: False Positive  TN: True Negative

    "TP of Class_1" is all Class_1 instances that are classified as Class_1.
    "TN of Class_1" is all non-Class_1 instances that are not classified as Class_1.
    "FP of Class_1" is all non-Class_1 instances that are classified as Class_1.
    "FN of Class_1" is all Class_1 instances that are not classified as Class_1.
    # https://www.researchgate.net/post
    /How_do_you_measure_specificity_and_sensitivity_in_a_multiple_class_classification_problem

    --> Input matrix
             | 0 Prediction | 1 Prediction | 2 Prediction | .....
    0 Class  |     T0       |     ..       |      ..      |
    1 Class  |     ..       |     T1       |      ..      |
    2 Class  |     ..       |     ..       |      T2      |

    --> Output matrix

                    | Positive Prediction | Negative Prediction
    Positive Class  |       TP            |       FN
    Negative Class  |       FP            |       TN

    :param confusion_mtr: 10 class confusion matrix designed for movielens
    :param class_i: index of the class we are interested in(0-9)
    :return: TP, FN, FP, TN
    """
    length = len(confusion_mtr)

    TP = confusion_mtr[class_i][class_i]

    actual_class_i_count = 0
    for i in range(length):  # sum of the row
      actual_class_i_count += confusion_mtr[class_i][i]
    FN = actual_class_i_count - TP

    predicted_class_i_count = 0
    for i in range(length):  # sum of the column
      predicted_class_i_count += confusion_mtr[i][class_i]
    FP = predicted_class_i_count - TP

    # sum of matrix
    sum_of_matrix = np.sum(confusion_mtr)
    # TN is found by summing up all values except the row and column of the class
    TN = sum_of_matrix - predicted_class_i_count - actual_class_i_count - TP

    return TP, FN, FP, TN

  @staticmethod
  def confusion_matrix(predictions):
    """
    Create confusion matrix and then return TP, FN, FP, TN

    0 Class: 0.5
    1 Class: 1
    2 Class: 1.5
    3 Class: 2
    4 Class: 2.5
    5 Class: 3
    6 Class: 3.5
    7 Class: 4
    8 Class: 4.5
    9 Class: 5

    T0: True 0
    F0: False 0
    T1: True 1
    F1: False 1
    ...

             | 0 Prediction | 1 Prediction | 2 Prediction | .....
    0 Class  |     T0       |     ..       |      ..      |
    1 Class  |     ..       |     T1       |      ..      |
    2 Class  |     ..       |     ..       |      T2      |
    ...
    """
    # Create multiclass confusion matrix

    conf_mtr = np.zeros((10, 10))

    for prediction in predictions:
      predicted = Accuracy.half_round_rating(prediction[0])
      actual = Accuracy.half_round_rating(prediction[1])

      predicted_class_index = int((predicted * 2) - 1)
      actual_class_index = int((actual * 2) - 1)

      conf_mtr[actual_class_index][predicted_class_index] += 1

    return conf_mtr

  @staticmethod
  def threshold_confusion_matrix(predictions):
    """
    Create confusion matrix and then return TP, FN, FP, TN

    Positive Class: 5
    Negative Class: 0.5

    TP: True Positive
    TN: True Negative
    FP: False Positive
    FN: False Negative

                    | Positive Prediction | Negative Prediction
    Positive Class  |       TP            |       FN
    Negative Class  |       FP            |       TN

    """
    # Create confusion matrix
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for prediction in predictions:
      predicted = Accuracy.threshold_round_rating(prediction[0])
      actual = Accuracy.threshold_round_rating(prediction[1])
      if predicted == 5 and actual == 5:
        TP += 1
      elif predicted == 5 and actual == 0.5:
        FP += 1
      elif predicted == 0.5 and actual == 0.5:
        TN += 1
      elif predicted == 0.5 and actual == 5:
        FN += 1
    return TP, FN, FP, TN

  @staticmethod
  def half_round_rating(rating):
    """
    Round ratings to the closest match in the movielens dataset
    For ex.
      ratings between 2 and 2.25 -> round to 2
      ratings between 2.25 and 2.5 -> round to 2.5
      ratings between 2.5 and 2.75 -> round to 2.5
      ratings between 2.75 and 3 -> round to 3

    """
    floor_value = math.floor(rating)
    if rating > floor_value + 0.75:
      return floor_value + 1
    elif rating > floor_value + 0.5 or rating > floor_value + 0.25:
      return floor_value + 0.5
    else:
      return floor_value

  @staticmethod
  def threshold_round_rating(rating):
    """
    Round ratings to the closest match in threshold fashion
      ratings between 0.5 and 3.5 -> round to 0.5
      ratings between 3.5 and 5 -> round to 5
    """
    if 0.5 <= rating < 3.5:
      return 0.5
    elif 3.5 <= rating <= 5:
      return 5
    else:
      return 0