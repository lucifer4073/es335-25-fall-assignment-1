import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tsfel
import warnings
warnings.filterwarnings('ignore')

def solve_problem4_correlation_analysis_fixed():
    # STEP 1: LOAD DATA
    combined_train_path = './Combined/Train'
    activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

    def load_and_standardize_data(data_path, activities, target_length=500):
        X_list = []
        y_list = []
        for activity_idx, activity in enumerate(activities):
            activity_path = os.path.join(data_path, activity)
            if not os.path.exists(activity_path):
                continue
            csv_files = [f for f in os.listdir(activity_path) if f.endswith('.csv')]
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
                        continue
                    X_list.append(standardized_data)
                    y_list.append(activity_idx)
                except:
                    continue
        return np.array(X_list), np.array(y_list)

    # Load data
    X_train, y_train = load_and_standardize_data(combined_train_path, activities)
    print(f"Loaded {X_train.shape[0]} training samples")

    # STEP 2: EXTRACT FEATURES USING ACTUAL TSFEL (Updated)
    def extract_tsfel_features(X):
        print("Extracting TSFEL features for correlation analysis...")
        
        cfg = tsfel.get_features_by_domain()
        features = []
        feature_names = []
        failed_count = 0
        
        for i, sample in enumerate(X):
            if i % 20 == 0:
                print(f"  Processing sample {i+1}/{len(X)}")
            
            try:
                # Create DataFrame for TSFEL
                df_sample = pd.DataFrame(sample, columns=['acc_x', 'acc_y', 'acc_z'])
                
                # Extract TSFEL features
                sample_features = tsfel.time_series_features_extractor(
                    cfg, df_sample, fs=50, verbose=0
                )
                
                if sample_features.shape[1] > 0:
                    # Store feature names (only once)
                    if len(feature_names) == 0:
                        feature_names = [col[:15] for col in sample_features.columns]  # Truncate for readability
                    
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
                    
                    if len(feature_names) == 0:
                        for axis_name in ['X', 'Y', 'Z']:
                            for feat in ['Mean', 'Std', 'Var', 'Max', 'Min', 'Med']:
                                feature_names.append(f'{axis_name}_{feat}')
                    
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
        return features_array, feature_names

    def extract_dataset_features(X):
        """Extract UCI-HAR style features"""
        features = []
        feature_names = []
        
        for sample in X:
            sample_features = []
            
            # Body acceleration
            gravity = np.mean(sample, axis=0)
            body_acc = sample - gravity
            
            for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
                orig_data = sample[:, axis_idx]
                body_data = body_acc[:, axis_idx]
                
                # Body and gravity features
                sample_features.extend([
                    np.mean(body_data), np.std(body_data),
                    gravity[axis_idx], np.std(orig_data - body_data)
                ])
                
                if len(features) == 0:
                    feature_names.extend([
                        f'Body{axis_name}_M', f'Body{axis_name}_S',
                        f'Grav{axis_name}_M', f'Grav{axis_name}_S'
                    ])
                
                # Jerk features
                if len(body_data) > 1:
                    jerk = np.diff(body_data)
                    sample_features.extend([np.mean(jerk), np.std(jerk)])
                else:
                    sample_features.extend([0.0, 0.0])
                
                if len(features) == 0:
                    feature_names.extend([f'Jerk{axis_name}_M', f'Jerk{axis_name}_S'])
            
            # Magnitude features
            body_mag = np.sqrt(np.sum(body_acc**2, axis=1))
            total_mag = np.sqrt(np.sum(sample**2, axis=1))
            sample_features.extend([
                np.mean(body_mag), np.std(body_mag),
                np.mean(total_mag), np.std(total_mag)
            ])
            
            if len(features) == 0:
                feature_names.extend(['BodyMag_M', 'BodyMag_S', 'TotMag_M', 'TotMag_S'])
            
            features.append(sample_features)
        
        return np.array(features), feature_names

    # Extract features
    tsfel_features, tsfel_names = extract_tsfel_features(X_train)
    dataset_features, dataset_names = extract_dataset_features(X_train)

    # Handle NaN values
    tsfel_features = np.nan_to_num(tsfel_features, nan=0.0, posinf=0.0, neginf=0.0)
    dataset_features = np.nan_to_num(dataset_features, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"TSFEL features: {tsfel_features.shape[1]}")
    print(f"Dataset features: {dataset_features.shape[1]}")

    # STEP 3: CALCULATE CORRELATION MATRICES
    tsfel_corr = np.corrcoef(tsfel_features.T)
    dataset_corr = np.corrcoef(dataset_features.T)

    # STEP 4: CREATE CLEAN VISUALIZATION
    print("\nCreating clean correlation matrix visualization...")

    # Create figure with larger size
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    # TSFEL correlation heatmap (no labels to avoid clutter)
    sns.heatmap(tsfel_corr,
                ax=axes[0],
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                cbar=True,
                square=True,
                xticklabels=False,  # Remove x labels
                yticklabels=False)  # Remove y labels

    axes[0].set_title(f'TSFEL Features Correlation Matrix\n({tsfel_features.shape[1]} features)',
                     fontsize=16, fontweight='bold', pad=20)
    axes[0].set_xlabel('Feature Index', fontsize=12)
    axes[0].set_ylabel('Feature Index', fontsize=12)

    # Dataset correlation heatmap (with simplified labels)
    x_labels = [dataset_names[i] if i % 3 == 0 else '' for i in range(len(dataset_names))]
    y_labels = [dataset_names[i] if i % 3 == 0 else '' for i in range(len(dataset_names))]

    sns.heatmap(dataset_corr,
                ax=axes[1],
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                cbar=True,
                square=True,
                xticklabels=x_labels,
                yticklabels=y_labels)

    axes[1].set_title(f'Dataset-style Features Correlation Matrix\n({dataset_features.shape[1]} features)',
                     fontsize=16, fontweight='bold', pad=20)
    axes[1].set_xlabel('Feature Index', fontsize=12)
    axes[1].set_ylabel('Feature Index', fontsize=12)

    # Rotate labels for better readability
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=10)
    plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=10)

    # Add overall title
    fig.suptitle('Feature Correlation Analysis for Human Activity Recognition',
                 fontsize=18, fontweight='bold', y=0.95)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # Save with high quality
    plt.savefig('EDA-4-result.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Correlation matrices saved as 'EDA-4-result.png'")

    # STEP 5: ANALYZE HIGH CORRELATIONS
    def find_high_correlations(corr_matrix, feature_names, threshold=0.8):
        high_corr_pairs = []
        n = corr_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if abs(corr_matrix[i, j]) >= threshold:
                    high_corr_pairs.append({
                        'idx1': i, 'idx2': j,
                        'name1': feature_names[i], 'name2': feature_names[j],
                        'correlation': corr_matrix[i, j]
                    })
        return high_corr_pairs

    # Find high correlations
    threshold = 0.8
    tsfel_high_corr = find_high_correlations(tsfel_corr, tsfel_names, threshold)
    dataset_high_corr = find_high_correlations(dataset_corr, dataset_names, threshold)

    print(f"\nHIGH CORRELATION ANALYSIS (threshold ≥ {threshold}):")
    print(f"TSFEL features: {len(tsfel_high_corr)} highly correlated pairs")
    print(f"Dataset features: {len(dataset_high_corr)} highly correlated pairs")

    # Show top correlations
    if tsfel_high_corr:
        print(f"\nTop TSFEL correlations:")
        for i, pair in enumerate(sorted(tsfel_high_corr, key=lambda x: abs(x['correlation']), reverse=True)[:5]):
            print(f" {i+1}. {pair['name1']} ↔ {pair['name2']}: r = {pair['correlation']:.3f}")

    if dataset_high_corr:
        print(f"\nTop Dataset correlations:")
        for i, pair in enumerate(sorted(dataset_high_corr, key=lambda x: abs(x['correlation']), reverse=True)[:5]):
            print(f" {i+1}. {pair['name1']} ↔ {pair['name2']}: r = {pair['correlation']:.3f}")

    # Calculate redundancy rates
    total_tsfel_pairs = len(tsfel_names) * (len(tsfel_names) - 1) // 2
    total_dataset_pairs = len(dataset_names) * (len(dataset_names) - 1) // 2
    tsfel_redundancy = len(tsfel_high_corr) / total_tsfel_pairs * 100 if total_tsfel_pairs > 0 else 0
    dataset_redundancy = len(dataset_high_corr) / total_dataset_pairs * 100 if total_dataset_pairs > 0 else 0

    print(f"\nREDUNDANCY ANALYSIS:")
    print(f"TSFEL features redundancy rate: {tsfel_redundancy:.2f}%")
    print(f"Dataset features redundancy rate: {dataset_redundancy:.2f}%")

    # Final conclusion
    print(f"\nCONCLUSION:")
    if max(len(tsfel_high_corr), len(dataset_high_corr)) > 0:
        print("YES, redundant features are present:")
        print(f"• TSFEL features: {len(tsfel_high_corr)} redundant pairs")
        print(f"• Dataset features: {len(dataset_high_corr)} redundant pairs")
        print("• Recommendation: Apply feature selection techniques")
    else:
        print("NO significant redundancy detected in both feature sets")

    print(f"\nPROBLEM 4 COMPLETED SUCCESSFULLY!")

    return {
        'tsfel_corr': tsfel_corr,
        'dataset_corr': dataset_corr,
        'tsfel_high_corr': tsfel_high_corr,
        'dataset_high_corr': dataset_high_corr,
        'redundancy_rates': (tsfel_redundancy, dataset_redundancy)
    }

# Execute the updated version
if __name__ == "__main__":
    results = solve_problem4_correlation_analysis_fixed()
