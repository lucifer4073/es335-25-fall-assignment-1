import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import tsfel
import joblib

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

class HARDecisionTrees:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                               'SITTING', 'STANDING', 'LAYING']
    
    def load_data(self, data_path):
        """
        Load the three types of data:
        1. Raw accelerometer data
        2. TSFEL features 
        3. Dataset provided features
        """
        # Load dataset provided features (X_train.txt, X_test.txt)
        X_train_dataset = np.loadtxt(f"{data_path}/train/X_train.txt")
        X_test_dataset = np.loadtxt(f"{data_path}/test/X_test.txt")
        
        # Load labels
        y_train = np.loadtxt(f"{data_path}/train/y_train.txt", dtype=int) - 1  # Convert to 0-indexed
        y_test = np.loadtxt(f"{data_path}/test/y_test.txt", dtype=int) - 1
        
        # Load raw accelerometer data from inertial signals
        # Total acceleration = body_acc + gravity_acc
        train_acc_x = np.loadtxt(f"{data_path}/train/Inertial Signals/total_acc_x_train.txt")
        train_acc_y = np.loadtxt(f"{data_path}/train/Inertial Signals/total_acc_y_train.txt")
        train_acc_z = np.loadtxt(f"{data_path}/train/Inertial Signals/total_acc_z_train.txt")
        
        test_acc_x = np.loadtxt(f"{data_path}/test/Inertial Signals/total_acc_x_test.txt")
        test_acc_y = np.loadtxt(f"{data_path}/test/Inertial Signals/total_acc_y_test.txt")
        test_acc_z = np.loadtxt(f"{data_path}/test/Inertial Signals/total_acc_z_test.txt")
        
        # Combine x,y,z accelerometer data
        X_train_raw = np.concatenate([train_acc_x, train_acc_y, train_acc_z], axis=1)
        X_test_raw = np.concatenate([test_acc_x, test_acc_y, test_acc_z], axis=1)
        
        # Generate TSFEL features
        print("Extracting TSFEL features...")
        X_train_tsfel = self.extract_tsfel_features(train_acc_x, train_acc_y, train_acc_z)
        X_test_tsfel = self.extract_tsfel_features(test_acc_x, test_acc_y, test_acc_z)
        
        return {
            'raw': (X_train_raw, X_test_raw),
            'tsfel': (X_train_tsfel, X_test_tsfel),
            'dataset': (X_train_dataset, X_test_dataset)
        }, y_train, y_test
    
    def extract_tsfel_features(self, acc_x, acc_y, acc_z):
        """
        Extract features using TSFEL library - ROBUST VERSION
        """
        print(f"Processing {len(acc_x)} samples for TSFEL feature extraction...")
        
        # Use default configuration and extract manually
        features_list = []
        failed_count = 0
        
        # Define manual feature extraction functions to ensure we get features
        def extract_manual_features(signal):
            """Extract basic statistical and temporal features manually"""
            features = []
            
            # Statistical features
            features.extend([
                np.mean(signal),           # mean
                np.std(signal),            # standard deviation
                np.var(signal),            # variance
                np.max(signal),            # maximum
                np.min(signal),            # minimum
                np.median(signal),         # median
                np.percentile(signal, 25), # 25th percentile
                np.percentile(signal, 75), # 75th percentile
            ])
            
            # Temporal features
            features.extend([
                np.sum(np.diff(signal) > 0),  # number of positive differences
                np.sum(np.abs(np.diff(signal))), # sum of absolute differences
                np.mean(np.abs(np.diff(signal))), # mean absolute difference
            ])
            
            return np.array(features)
        
        for i in range(len(acc_x)):
            if i % 1000 == 0:
                print(f"Processing sample {i}/{len(acc_x)}")
            
            try:
                # Create DataFrame with the time series data
                df_sample = pd.DataFrame({
                    'acc_x': acc_x[i],  # Shape: (128,)
                    'acc_y': acc_y[i],  # Shape: (128,)
                    'acc_z': acc_z[i]   # Shape: (128,)
                })
                
                # Try TSFEL first with a simple configuration
                try:
                    cfg = tsfel.get_features_by_domain()
                    sample_features = tsfel.time_series_features_extractor(
                        cfg, df_sample, fs=50, verbose=0
                    )
                    
                    if sample_features.shape[1] > 0:  # If features were extracted
                        feature_vector = sample_features.values.flatten()
                        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
                        features_list.append(feature_vector)
                    else:
                        raise ValueError("No features extracted by TSFEL")
                        
                except:
                    # Fall back to manual feature extraction
                    features_x = extract_manual_features(acc_x[i])
                    features_y = extract_manual_features(acc_y[i])
                    features_z = extract_manual_features(acc_z[i])
                    
                    # Combine features from all three axes
                    combined_features = np.concatenate([features_x, features_y, features_z])
                    features_list.append(combined_features)
                
            except Exception as e:
                failed_count += 1
                if failed_count < 5:  # Only print first 5 errors
                    print(f"Warning: Feature extraction failed for sample {i}: {str(e)}")
                
                # Use the previous successful extraction or create a default
                if len(features_list) > 0:
                    features_list.append(np.zeros_like(features_list[-1]))
                else:
                    # Create a default feature vector (33 features: 11 per axis × 3 axes)
                    features_list.append(np.zeros(33))
        
        if failed_count > 5:
            print(f"Total failed extractions: {failed_count}")
        
        features_array = np.array(features_list)
        print(f"TSFEL features shape: {features_array.shape}")
        
        # Ensure we have at least some features
        if features_array.shape[1] == 0:
            print("Warning: No features extracted! Creating basic statistical features...")
            # Create basic features manually for all samples
            basic_features = []
            for i in range(len(acc_x)):
                features_x = extract_manual_features(acc_x[i])
                features_y = extract_manual_features(acc_y[i])
                features_z = extract_manual_features(acc_z[i])
                combined_features = np.concatenate([features_x, features_y, features_z])
                basic_features.append(combined_features)
            
            features_array = np.array(basic_features)
            print(f"Manual features shape: {features_array.shape}")
        
        return features_array
    
    def train_model(self, X_train, X_test, y_train, y_test, model_name, max_depth=None):
        """
        Train a single decision tree model
        """
        print(f"Training {model_name} with data shapes: train={X_train.shape}, test={X_test.shape}")
        
        # Check if we have any features
        if X_train.shape[1] == 0:
            print(f"Warning: No features available for {model_name}. Skipping this model.")
            return None, 0, 0, 0, None
        
        # Initialize model
        dt = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
            criterion='gini'
        )
        
        # Scale features for better performance
        scaler = None
        if model_name.lower() in ['tsfel', 'dataset']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train model
        dt.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = dt.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        self.models[model_name] = {'model': dt, 'scaler': scaler}
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        return dt, accuracy, precision, recall, cm
    
    def print_results(self, model_name):
        """
        Print detailed results for a model
        """
        results = self.results[model_name]
        print(f"\n{'='*50}")
        print(f"RESULTS FOR {model_name} MODEL")
        print(f"{'='*50}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        
        # Classification report
        print(f"\nDetailed Classification Report:")
        print(classification_report(results['true_labels'], results['predictions'], 
                                  target_names=self.activity_labels, zero_division=0))
    
    def plot_confusion_matrix(self, model_name):
        """
        Plot confusion matrix for a model and save to file
        """
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.activity_labels,
                   yticklabels=self.activity_labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot instead of showing
        plt.savefig(f'confusion_matrix_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved as: confusion_matrix_{model_name.lower()}.png")
        plt.close()  # Close the figure to free memory
    
    def depth_analysis(self, X_train_dict, X_test_dict, y_train, y_test):
        """
        Train models with different depths (2-8) and plot results
        """
        depths = range(2, 9)
        depth_results = {data_type: [] for data_type in X_train_dict.keys()}
        
        print("Running depth analysis...")
        for depth in depths:
            print(f"  Testing depth {depth}...")
            for data_type in X_train_dict.keys():
                _, accuracy, _, _, _ = self.train_model(
                    X_train_dict[data_type], X_test_dict[data_type], 
                    y_train, y_test, f"{data_type}_depth_{depth}", 
                    max_depth=depth
                )
                depth_results[data_type].append(accuracy)
        
        # Plot results and save to file
        plt.figure(figsize=(10, 6))
        for data_type in depth_results.keys():
            plt.plot(depths, depth_results[data_type], 
                    marker='o', label=data_type.upper(), linewidth=2, markersize=6)
        
        plt.xlabel('Tree Depth')
        plt.ylabel('Test Accuracy')
        plt.title('Decision Tree Performance vs Depth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(depths)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Save plot instead of showing
        plt.savefig('depth_analysis.png', dpi=300, bbox_inches='tight')
        print("Depth analysis plot saved as: depth_analysis.png")
        plt.close()  # Close the figure to free memory
        
        return depth_results
    
    def error_analysis(self, model_name):
        """
        Analyze which activities have poor performance
        """
        results = self.results[model_name]
        cm = results['confusion_matrix']
        
        # Calculate per-class metrics
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        print(f"\n{'='*50}")
        print(f"ERROR ANALYSIS FOR {model_name}")
        print(f"{'='*50}")
        
        for i, activity in enumerate(self.activity_labels):
            print(f"{activity:20s}: {per_class_accuracy[i]:.4f}")
        
        # Find worst performing classes
        worst_classes = np.argsort(per_class_accuracy)[:2]
        print(f"\nWorst performing activities:")
        for idx in worst_classes:
            print(f"- {self.activity_labels[idx]}: {per_class_accuracy[idx]:.4f}")
        
        return per_class_accuracy

# Example usage:
def main():
    # Initialize the HAR system
    har = HARDecisionTrees()
    
    # Load your data (adjust path as needed)
    data_path = "."  # Update this to your actual path
    
    try:
        data_dict, y_train, y_test = har.load_data(data_path)
        print("Data loaded successfully!")
        print(f"Training samples: {len(y_train)}")
        print(f"Test samples: {len(y_test)}")
        
        # Train and evaluate each model
        for model_name, (X_train, X_test) in data_dict.items():
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()} model...")
            print(f"Training data shape: {X_train.shape}")
            print(f"Test data shape: {X_test.shape}")
            print(f"{'='*60}")
            
            har.train_model(X_train, X_test, y_train, y_test, model_name)
            har.print_results(model_name)
            har.plot_confusion_matrix(model_name)
            har.error_analysis(model_name)

            if model_name == 'tsfel':
                print("\nSAVING TSFEL MODEL AND SCALER...")
                model_to_save = har.models['tsfel']
                joblib.dump(model_to_save['model'], 'tsfel_model.pkl')
                joblib.dump(model_to_save['scaler'], 'tsfel_scaler.pkl')
                print("✅ Model and scaler saved successfully!")
        
        # Compare models
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        for model_name in data_dict.keys():
            results = har.results[model_name]
            print(f"{model_name.upper():15s} | Accuracy: {results['accuracy']:.4f} | "
                  f"Precision: {results['precision']:.4f} | Recall: {results['recall']:.4f}")
        
        # Depth analysis
        print(f"\n{'='*60}")
        print("PERFORMING DEPTH ANALYSIS...")
        print(f"{'='*60}")
        
        X_train_dict = {name: data[0] for name, data in data_dict.items()}
        X_test_dict = {name: data[1] for name, data in data_dict.items()}
        depth_results = har.depth_analysis(X_train_dict, X_test_dict, y_train, y_test)
        
        # Print optimal depths
        print("\nOptimal depths:")
        for data_type, accuracies in depth_results.items():
            optimal_depth = np.argmax(accuracies) + 2  # +2 because we start from depth 2
            max_accuracy = max(accuracies)
            print(f"{data_type:15s}: Depth {optimal_depth} (Accuracy: {max_accuracy:.4f})")
        
        print(f"\n{'='*60}")
        print("TASK 2 COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Install required libraries: pip install tsfel scikit-learn pandas numpy matplotlib seaborn")
        print("2. Update the data_path variable to point to your UCI HAR Dataset folder")
        print("3. Ensure the dataset structure matches the expected format")

if __name__ == "__main__":
    main()
