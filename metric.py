def f1_score_tensor(y_pred, y_true, positive_list):
    pred_list = y_pred.view(-1)
    true_list = y_true.view(-1)
    assert len(pred_list) == len(true_list)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index, element in enumerate(pred_list):
        if element == true_list[index]:
            if element in positive_list:
                TP += 1
            else:
                TN += 1
        else:
            if element in positive_list:
                FP += 1
            else:
                FN += 1
    assert len(pred_list) == (TP + TN + FP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def f1_score_list(y_pred, y_true, positive_list):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i, item in enumerate(y_pred):
        for j, element in enumerate(item):
            if element == y_true[i][j]:
                if element in positive_list:
                    TP += 1
                else:
                    TN += 1
            else:
                if element in positive_list:
                    FP += 1
                else:
                    FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
