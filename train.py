import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. Start the 'Flight Recorder'
with mlflow.start_run():
    # 2. Define the AI Model (The Software)
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100)
    model.fit(iris.data, iris.target)
    
    # 3. Log the 'Evidence' (DevOps Tracking)
    mlflow.log_param("model_type", "RandomForest")
    mlflow.sklearn.log_model(model, "iris_model")
    
    print("Step 1 Complete: Model trained and logged!")