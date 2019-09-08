def scale_confusion_matrix(confm):
    return (confm.transpose() / confm.sum(1)).transpose()
