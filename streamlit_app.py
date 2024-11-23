
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# File paths for datasets
file_paths = {
    'Kolkata': 'Kolkata.csv',
    'Goa': 'Goa.csv',
    'Meghalaya': 'Meghalaya.csv',
    'Mizoram': 'Mizoram.csv'
}

# Load and combine datasets
dataframes = {location: pd.read_csv(path) for location, path in file_paths.items()}

# Function to preprocess individual datasets
def preprocess_dataset(df, location):
    # Handle missing values
    df['Average'] = df['Average'].fillna(df['Average'].mean())
    
    # Add a location column to differentiate data
    df['Location'] = location

    # Add drought classification
    drought_threshold = 3.0
    df['Drought'] = df['Average'].apply(lambda x: 1 if x < drought_threshold else 0)
    
    return df

# Preprocess all datasets
processed_dataframes = {loc: preprocess_dataset(df.copy(), loc) for loc, df in dataframes.items()}
combined_data = pd.concat(processed_dataframes.values(), ignore_index=True)

# Upsample minority class to handle imbalance
majority = combined_data[combined_data['Drought'] == 0]
minority = combined_data[combined_data['Drought'] == 1]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

balanced_data = pd.concat([majority, minority_upsampled])

# Prepare features and target
X = balanced_data[['Average']]
y = balanced_data['Drought']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=100)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=1)

print("Accuracy:", accuracy)

# Prediction function for user input with probability
def predict_drought(location, avg_rainfall):
    if location not in file_paths.keys():
        return f"Invalid location: {location}. Available locations are {list(file_paths.keys())}."
    
    # Prepare the input sample
    sample = pd.DataFrame({'Average': [avg_rainfall]})
    scaled_sample = scaler.transform(sample)
    
    # Predict drought probabilities
    prob = model.predict_proba(scaled_sample)
    drought_probability = prob[0][1]  # Probability of class 1 (drought)
    
    return f"The probability of drought in {location} for the given average rainfall of {avg_rainfall:.2f} mm is {drought_probability:.4f}."

# Example usage
if __name__ == "__main__":
    location_to_predict = "Goa"
    average_rainfall_to_predict = 2  # Example average rainfall value
    
    print("\nPrediction:")
    print(predict_drought(location_to_predict, average_rainfall_to_predict))
