from tpot import TPOTClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TPOT
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)

# Fit the model
tpot.fit(X_train, y_train)

# Evaluate
print("Test accuracy:", tpot.score(X_test, y_test))

# Export the best pipeline
tpot.export('tpot_pipeline.py')
