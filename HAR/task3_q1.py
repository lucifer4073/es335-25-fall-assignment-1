import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tsfel
import warnings
warnings.filterwarnings('ignore')

def solve_task3_question1_linear_accel():  
    activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                  'SITTING', 'STANDING', 'LAYING']
    
    def load_and_process_linear_accel_data(data_dir, activities):
        """Load and process personal Linear Accelerometer CSV files"""
        X_personal = []
        y_personal = []
        file_info = []
        
        print(f"Loading Linear Accelerometer data from: {data_dir}")
        
        for activity_idx, activity in enumerate(activities):
            activity_path = os.path.join(data_dir, activity)
            
            if not os.path.exists(activity_path):
                print(f"Activity folder not found: {activity}")
                continue
            
            csv_files = [f for f in os.listdir(activity_path) if f.endswith('.csv')]
            print(f"  {activity}: {len(csv_files)} files found")
            
            for csv_file in csv_files:
                try:
                    file_path = os.path.join(activity_path, csv_file)
                    
                    # Read CSV, skip comment lines starting with #
                    df = pd.read_csv(file_path, comment='#')
                    print(f"    {csv_file}: {len(df)} samples (~50Hz)")
                    
                    # Extract Linear Accelerometer data (ax, ay, az - columns 1,2,3)
                    if df.shape[1] >= 4 and 'ax (m/s^2)' in df.columns:
                        accel_data = df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']].values
                    else:
                        print(f"Unexpected format in {csv_file}")
                        continue
                    
                    # Data is already ~50Hz, so minimal processing needed
                    print(f"    Linear Accel Data: {len(accel_data)} samples at ~50Hz")
                    
                    # Create 10-second segments (500 samples at 50Hz)
                    target_samples = 500
                    
                    if len(accel_data) >= target_samples + 100:
                        # Remove artifacts from start/end
                        start_trim = 50
                        end_trim = 50
                        middle_data = accel_data[start_trim:-end_trim]
                        
                        if len(middle_data) >= target_samples:
                            start_idx = (len(middle_data) - target_samples) // 2
                            segment = middle_data[start_idx:start_idx + target_samples]
                        else:
                            segment = middle_data[:target_samples]
                    else:
                        segment = accel_data[:target_samples]
                    
                    # Pad if necessary
                    if len(segment) < target_samples:
                        padding_needed = target_samples - len(segment)
                        padding = np.tile(segment[-1], (padding_needed, 1))
                        segment = np.vstack([segment, padding])
                    
                    print(f"    Final segment: {len(segment)} samples (10s at 50Hz)")
                    
                    X_personal.append(segment)
                    y_personal.append(activity_idx)
                    file_info.append(f"{activity}/{csv_file}")
                    
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
        
        return np.array(X_personal), np.array(y_personal), file_info
    
    def extract_tsfel_features_for_uci_compatibility(X_segments):
        """
        Extract TSFEL features compatible with UCI model
        Using the same method as har_decision_trees.py for maximum compatibility
        """
        print("Extracting TSFEL features (UCI-compatible method)...")
        
        # Separate X, Y, Z axes from segments for TSFEL processing
        acc_x = X_segments[:, :, 0]  # All samples, all timepoints, X-axis
        acc_y = X_segments[:, :, 1]  # All samples, all timepoints, Y-axis  
        acc_z = X_segments[:, :, 2]  # All samples, all timepoints, Z-axis
        
        print(f"Processing {len(acc_x)} samples for TSFEL feature extraction...")
        
        features_list = []
        failed_count = 0
        
        def extract_manual_features(signal):
            """Extract basic statistical and temporal features manually"""
            features = []
            # Statistical features
            features.extend([
                np.mean(signal),
                np.std(signal),
                np.var(signal),
                np.max(signal),
                np.min(signal),
                np.median(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
            ])
            # Temporal features  
            features.extend([
                np.sum(np.diff(signal) > 0),
                np.sum(np.abs(np.diff(signal))),
                np.mean(np.abs(np.diff(signal))),
            ])
            return np.array(features)

        for i in range(len(acc_x)):
            if i % 5 == 0:
                print(f"    Processing sample {i+1}/{len(acc_x)}")
            
            try:
                # Create DataFrame exactly like in har_decision_trees.py
                df_sample = pd.DataFrame({
                    'acc_x': acc_x[i],  # Shape: (500,) 
                    'acc_y': acc_y[i],  # Shape: (500,)
                    'acc_z': acc_z[i]   # Shape: (500,)
                })
                
                # Try TSFEL first with same configuration as training
                try:
                    cfg = tsfel.get_features_by_domain()
                    sample_features = tsfel.time_series_features_extractor(
                        cfg, df_sample, fs=50, verbose=0
                    )
                    
                    if sample_features.shape[1] > 0:
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
                    combined_features = np.concatenate([features_x, features_y, features_z])
                    features_list.append(combined_features)
                    
            except Exception as e:
                failed_count += 1
                if failed_count < 3:
                    print(f"    Warning: Feature extraction failed for sample {i}: {str(e)}")
                
                # Use previous successful extraction or create default
                if len(features_list) > 0:
                    features_list.append(np.zeros_like(features_list[-1]))
                else:
                    features_list.append(np.zeros(33))  # Default 33 features
        
        features_array = np.array(features_list)
        print(f"TSFEL features extracted: {features_array.shape}")
        
        # Handle case where no features were extracted
        if features_array.shape[1] == 0:
            print("Falling back to manual feature extraction for all samples...")
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
    
    def predict_with_uci_model(X_personal_features, y_personal, activities, file_info):
        """Load UCI model and make predictions on Linear Accelerometer data"""
        print(f"\nLoading UCI-trained TSFEL model...")
        
        try:
            uci_model = joblib.load('tsfel_model.pkl')
            uci_scaler = joblib.load('tsfel_scaler.pkl')
            print("Successfully loaded UCI model and scaler")
            
            expected_features = uci_scaler.mean_.shape[0] if hasattr(uci_scaler, 'mean_') else 468
            actual_features = X_personal_features.shape[1] 
            
            print(f"   UCI model expects: {expected_features} features")
            print(f"   Linear Accel data has: {actual_features} features")
            
            if expected_features != actual_features:
                print(f"⚠️FEATURE DIMENSION MISMATCH!")
                print(f"   Expected: {expected_features}, Got: {actual_features}")
                
                if actual_features < expected_features:
                    # Pad with zeros
                    padding = np.zeros((X_personal_features.shape[0], expected_features - actual_features))
                    X_personal_features = np.hstack([X_personal_features, padding])
                    print(f"   Padded features to {X_personal_features.shape[1]}")
                else:
                    # Truncate
                    X_personal_features = X_personal_features[:, :expected_features]
                    print(f"   Truncated features to {X_personal_features.shape[1]}")
            
            # Scale and predict
            print("Scaling features...")
            X_scaled = uci_scaler.transform(X_personal_features)
            
            print("Making predictions...")
            y_pred = uci_model.predict(X_scaled)
            y_pred_proba = uci_model.predict_proba(X_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_personal, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_personal, y_pred, average='weighted', zero_division=0
            )
            
            print(f"\nRESULTS:")
            print("="*50)
            print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            
            # Classification report
            print(f"\nDETAILED REPORT:")
            print(classification_report(y_personal, y_pred, target_names=activities, digits=4, zero_division=0))
            
            # Confusion matrix
            cm = confusion_matrix(y_personal, y_pred)
            
            # Sample predictions
            print(f"\nPREDICTIONS:")
            max_proba = np.max(y_pred_proba, axis=1)
            for i, info in enumerate(file_info):
                true_act = activities[y_personal[i]]
                pred_act = activities[y_pred[i]]
                conf = max_proba[i]
                status = "✅" if y_personal[i] == y_pred[i] else "❌"
                print(f"   {status} {info:25} True: {true_act:15} → Pred: {pred_act:15} ({conf:.3f})")
            
            # Visualization
            plt.figure(figsize=(12, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=activities, yticklabels=activities)
            plt.title(f'UCI Model on Linear Accelerometer Data\nAccuracy: {accuracy:.3f} ({accuracy*100:.1f}%)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig('task3_q1_linear_accel_results.png', dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved as 'task3_q1_linear_accel_results.png'")
            plt.close()  # Prevent hanging
            
            # Performance analysis
            print(f"\nPERFORMANCE ANALYSIS:")
            if accuracy >= 0.7:
                print("   EXCELLENT: Great cross-domain performance with Linear Accelerometer!")
                print("   The switch to Linear Accelerometer data significantly improved compatibility")
            elif accuracy >= 0.5:
                print("   GOOD: Decent cross-domain performance")
                print("   Linear Accelerometer data shows better UCI compatibility than gForce")
            elif accuracy >= 0.3:
                print("   MODERATE: Some improvement over previous gForce results")
            else:
                print("   CHALLENGING: Still facing domain gap issues")
            
            return {
                'accuracy': accuracy,
                'precision': precision, 
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
        except FileNotFoundError:
            print("Model files not found!")
            print("   Make sure 'tsfel_model.pkl' and 'tsfel_scaler.pkl' exist")
            print("   Run har_decision_trees.py first to generate these files")
            return None
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # MAIN EXECUTION
    personal_data_path = './my_har_data'
    
    try:
        # Load personal Linear Accelerometer data
        X_personal, y_personal, file_info = load_and_process_linear_accel_data(
            personal_data_path, activities
        )
        
        if len(X_personal) == 0:
            print("No personal data found!")
            return None
        
        print(f"\nData Summary:")
        print(f"   Total samples: {len(X_personal)}")
        print(f"   Sample shape: {X_personal[0].shape}")
        print(f"   Data type: Linear Accelerometer (ax, ay, az in m/s²)")
        
        # Extract TSFEL features using UCI-compatible method
        X_features = extract_tsfel_features_for_uci_compatibility(X_personal)
        
        # Make predictions using UCI model
        results = predict_with_uci_model(X_features, y_personal, activities, file_info)
        
        if results:
            print(f"\nTASK 3 QUESTION 1 COMPLETED!")
            accuracy_pct = results['accuracy'] * 100
            print(f"   Final Accuracy: {results['accuracy']:.4f} ({accuracy_pct:.1f}%)")
            
            # Compare with previous gForce results (16.7%)
            improvement = (results['accuracy'] - 0.167) * 100
            print(f"   Improvement over gForce data: {improvement:+.1f} percentage points")
            
            if improvement > 0:
                print("   SUCCESS: Linear Accelerometer data performs better than gForce!")
            
            return results
        else:
            print("Prediction failed!")
            return None
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Execute the fixed solution
if __name__ == "__main__":
    results = solve_task3_question1_linear_accel()
    if results:
        print(f"\n🏁 SUCCESS! Linear Accelerometer Accuracy: {results['accuracy']:.4f}")
