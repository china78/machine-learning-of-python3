def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0]
    accuracy = sum(y_predict == y_true) / len(y_true)
    return accuracy
