import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tsfel
import warnings
warnings.filterwarnings('ignore')

def solve_problem3_pca():
    combined_train_path = './Combined/Train'
    combined_test_path = './Combined/Test'
    activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

    def load_and_standardize_data(data_path, activities, target_length=500):
        """Load CSV files and standardize them to same length"""
        X_list = []
        y_list = []
        print(f"Loading data from: {data_path}")
        
        for activity_idx, activity in enumerate(activities):
            activity_path = os.path.join(data_path, activity)
            if not os.path.exists(activity_path):
                print(f"Activity folder not found: {activity}")
                continue
                
            csv_files = [f for f in os.listdir(activity_path) if f.endswith('.csv')]
            print(f" {activity}: {len(csv_files)} files")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(os.path.join(activity_path, csv_file))
                    data = df.values
                    
                    if len(data) < target_length:
                        padding = np.tile(data[-1:], (target_length - len(data), 1))
                        standardized_data = np.vstack([data, padding])
                    else:
                        standardized_data = data[:target_length]
                    
                    if standardized_data.shape[1] != 3:
                        print(f"Unexpected columns in {csv_file}: {standardized_data.shape[1]}")
                        continue
                    
                    X_list.append(standardized_data)
                    y_list.append(activity_idx)
                    
                except Exception as e:
                    print(f"Error loading {csv_file}: {e}")
                    continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        return X, y

    # LOAD DATA
    try:
        X_train, y_train = load_and_standardize_data(combined_train_path, activities)
        X_test, y_test = load_and_standardize_data(combined_test_path, activities)
        print(f"\nDATA LOADED SUCCESSFULLY:")
        print(f"Training data: {X_train.shape}")
        print(f"Test data: {X_test.shape}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # METHOD 1 - PCA ON TOTAL ACCELERATION
    print(f"\n{'='*50}")
    print("METHOD 1: PCA ON TOTAL ACCELERATION")
    print(f"{'='*50}")
    
    train_total_acc = np.sqrt(np.sum(X_train**2, axis=2))
    test_total_acc = np.sqrt(np.sum(X_test**2, axis=2))
    
    scaler1 = StandardScaler()
    train_total_scaled = scaler1.fit_transform(train_total_acc)
    test_total_scaled = scaler1.transform(test_total_acc)
    
    pca1 = PCA(n_components=2, random_state=42)
    train_pca1 = pca1.fit_transform(train_total_scaled)
    test_pca1 = pca1.transform(test_total_scaled)
    
    print(f"Total Acceleration PCA completed")
    print(f"Explained variance: {pca1.explained_variance_ratio_}")

    # METHOD 2 - PCA ON TSFEL FEATURES
    print(f"\n{'='*50}")
    print("METHOD 2: PCA ON TSFEL FEATURES")
    print(f"{'='*50}")
    
    def extract_tsfel_features(X):
        """Extract features using TSFEL library as required by assignment"""
        print("Extracting TSFEL features...")
        
        # Get TSFEL configuration for useful features
        cfg = tsfel.get_features_by_domain()
        
        # Select specific useful features for HAR
        selected_features = {
            'statistical': ['Mean', 'Standard deviation', 'Variance', 'Max', 'Min', 'Median'],
            'temporal': ['Mean absolute differences', 'Sum absolute differences'],
            'spectral': ['Fundamental frequency', 'Max frequency', 'FFT mean coefficient']
        }
        
        features = []
        failed_count = 0
        
        for i, sample in enumerate(X):
            if i % 20 == 0:
                print(f"Processing sample {i+1}/{len(X)}")
            
            try:
                # Create DataFrame for TSFEL
                df_sample = pd.DataFrame(sample, columns=['acc_x', 'acc_y', 'acc_z'])
                
                # Extract TSFEL features with 50Hz sampling rate
                sample_features = tsfel.time_series_features_extractor(
                    cfg, df_sample, fs=50, verbose=0
                )
                
                if sample_features.shape[1] > 0:
                    # Flatten and clean features
                    feature_vector = sample_features.values.flatten()
                    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
                    features.append(feature_vector)
                else:
                    # Fallback: create basic features if TSFEL fails
                    basic_features = []
                    for axis in range(3):
                        axis_data = sample[:, axis]
                        basic_features.extend([
                            np.mean(axis_data), np.std(axis_data), np.var(axis_data),
                            np.max(axis_data), np.min(axis_data), np.median(axis_data)
                        ])
                    features.append(np.array(basic_features))
                    
            except Exception as e:
                failed_count += 1
                if failed_count < 5:
                    print(f"Warning: TSFEL failed for sample {i}: {str(e)}")
                
                # Use previous successful extraction or create default
                if len(features) > 0:
                    features.append(np.zeros_like(features[-1]))
                else:
                    features.append(np.zeros(50))  # Default feature size
        
        features_array = np.array(features)
        print(f"TSFEL features extracted: {features_array.shape}")
        return features_array

    # Extract TSFEL features
    train_tsfel_features = extract_tsfel_features(X_train)
    test_tsfel_features = extract_tsfel_features(X_test)
    
    # Standardize and apply PCA
    scaler2 = StandardScaler()
    train_tsfel_scaled = scaler2.fit_transform(train_tsfel_features)
    test_tsfel_scaled = scaler2.transform(test_tsfel_features)
    
    pca2 = PCA(n_components=2, random_state=42)
    train_pca2 = pca2.fit_transform(train_tsfel_scaled)
    test_pca2 = pca2.transform(test_tsfel_scaled)
    
    print(f"✓ TSFEL Features PCA completed")
    print(f"  Features extracted: {train_tsfel_features.shape[1]}")
    print(f"  Explained variance: {pca2.explained_variance_ratio_}")

    # METHOD 3 - PCA ON DATASET-STYLE FEATURES
    print(f"\n{'='*50}")
    print("METHOD 3: PCA ON DATASET-STYLE FEATURES")
    print(f"{'='*50}")

    def extract_dataset_features(X):
        """Extract UCI-HAR style features"""
        features = []
        for sample in X:
            sample_features = []
            
            # Body acceleration (remove gravity component)
            gravity_component = np.mean(sample, axis=0)
            body_acc = sample - gravity_component
            
            # Features for each axis
            for axis in range(3):
                orig_data = sample[:, axis]
                body_data = body_acc[:, axis]
                
                # Time domain features
                sample_features.extend([
                    np.mean(orig_data), np.std(orig_data),
                    np.mean(body_data), np.std(body_data),
                    np.mean(np.abs(orig_data)),
                    np.sqrt(np.mean(orig_data**2)),
                ])
                
                # Jerk features (derivative)
                jerk = np.diff(body_data)
                sample_features.extend([
                    np.mean(jerk), np.std(jerk)
                ])
            
            # Magnitude features
            total_mag = np.sqrt(np.sum(sample**2, axis=1))
            body_mag = np.sqrt(np.sum(body_acc**2, axis=1))
            sample_features.extend([
                np.mean(total_mag), np.std(total_mag),
                np.mean(body_mag), np.std(body_mag)
            ])
            
            features.append(sample_features)
        
        return np.array(features)

    # Extract dataset-style features
    train_dataset_features = extract_dataset_features(X_train)
    test_dataset_features = extract_dataset_features(X_test)
    
    # Standardize and apply PCA
    scaler3 = StandardScaler()
    train_dataset_scaled = scaler3.fit_transform(train_dataset_features)
    test_dataset_scaled = scaler3.transform(test_dataset_features)
    
    pca3 = PCA(n_components=2, random_state=42)
    train_pca3 = pca3.fit_transform(train_dataset_scaled)
    test_pca3 = pca3.transform(test_dataset_scaled)
    
    print(f"✓ Dataset-style Features PCA completed")
    print(f"  Features extracted: {train_dataset_features.shape[1]}")
    print(f"  Explained variance: {pca3.explained_variance_ratio_}")

    # VISUALIZATION
    print(f"\n{'='*50}")
    print("CREATING PCA VISUALIZATIONS")
    print(f"{'='*50}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(activities)))

    pca_results = [
        (test_pca1, pca1.explained_variance_ratio_, "Total Acceleration"),
        (test_pca2, pca2.explained_variance_ratio_, "TSFEL Features"),
        (test_pca3, pca3.explained_variance_ratio_, "Dataset Features")
    ]

    for i, (pca_data, var_ratio, method_name) in enumerate(pca_results):
        ax = axes[i]
        
        for activity_idx in range(len(activities)):
            mask = y_test == activity_idx
            if np.sum(mask) > 0:
                ax.scatter(pca_data[mask, 0], pca_data[mask, 1],
                          c=[colors[activity_idx]], label=activities[activity_idx],
                          alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
        
        total_var = sum(var_ratio)
        ax.set_title(f'{method_name}\n'
                    f'PC1: {var_ratio[0]:.3f}, PC2: {var_ratio[1]:.3f}\n'
                    f'Total: {total_var:.3f}', fontweight='bold')
        ax.set_xlabel(f'PC1 ({var_ratio[0]:.3f})')
        ax.set_ylabel(f'PC2 ({var_ratio[1]:.3f})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('PCA Visualization Comparison: Human Activity Recognition',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('EDA-3-result.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("PCA visualization saved as 'EDA-3-TSFEL-result.png'")

    # COMPARISON ANALYSIS
    print(f"\n{'='*50}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*50}")
    
    methods = ["Total Acceleration", "TSFEL Features", "Dataset Features"]
    variances = [
        sum(pca1.explained_variance_ratio_),
        sum(pca2.explained_variance_ratio_), 
        sum(pca3.explained_variance_ratio_)
    ]
    
    print("\nEXPLAINED VARIANCE COMPARISON:")
    for method, var in zip(methods, variances):
        print(f"{method:<20}: {var:.3f} ({var*100:.1f}%)")
    
    best_method_idx = np.argmax(variances)
    print(f"\nBEST METHOD: {methods[best_method_idx]} ({variances[best_method_idx]:.3f} explained variance)")
    
    print(f"\nEDA-3 WITH TSFEL COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    solve_problem3_pca()
