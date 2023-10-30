import numpy as np

class Metrics():
    def __init__(self, predicted, y):
        self.predicted = predicted
        self.y = y
        self.matrix = self.confusionMatrix()
        self.acc = self.accuracy()
        self.precision = self.precision()
        self.recall = self.recall()
        self.f1 = self.f1()

    
    def confusionMatrix(self):
        tp = np.sum(self.predicted == self.y)
        tn = np.sum(self.predicted != self.y)
        fp = sum(1 for true_label, predicted_label in zip(self.y, self.predicted)
                if predicted_label == 1 and true_label == 0)
        fn = sum(1 for true_label, predicted_label in zip(self.y, self.predicted)
                if predicted_label == 0 and true_label == 1)

        return [[tp, tn], [fp, fn]]

    def accuracy(self):
        correct_predictions = np.sum(self.predicted == self.y)
        total_predictions = len(self.y)
        accuracy = correct_predictions / total_predictions
        return accuracy

    def precision(self):
        tp = self.matrix[0][0]
        fp = self.matrix[1][0]

        return round(tp / (tp + fp), 4)


    def recall(self):
        tp = self.matrix[0][0]
        fn = self.matrix[1][1]

        return round(tp / (tp + fn), 4)


    def f1(self):
        return round((2 * self.precision * self.recall) / (self.precision + self.recall), 4)

