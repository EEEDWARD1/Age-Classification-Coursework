from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class AgeMLP:
  def __init__(self, hidden_layers=(256,128), max_iter=30):
    self.scaler = StandardScaler()
    self.model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=42
      )
  def fit(self, X_train, y_train):
    X_train_scaled = self.scaler.fit_transform(X_train)
    self.model.fit(X_train_scaled, y_train)

  def predict(self, X):
    X_scaled = self.scaler.transform(X)
    return self.model.predict(X_scaled)

  def evaluate(self, X, y):
    y_pred = self.predict(X)
    acc = accuracy_score(y, y_pred)

    print("Accuracy:", acc)
    print("\nClassification Report:\n")
    print(classification_report(y, y_pred))

    return acc
  
  def predict_image(self, img, extractor):
    features = extractor.extract(img)
    features = features.reshape(1, -1)
    return self.model.predict(self.scaler.transform(features))[0]