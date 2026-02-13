from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

class FallDetectorBaseline:

    def __init__(self, threshold, duration):
        self.threshold = threshold
        self.duration = duration

        def predict(self, video_path):
            return 0