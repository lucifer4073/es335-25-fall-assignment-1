import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_linear_acceleration_static_vs_dynamic():
    """
    Analyze linear acceleration to determine if ML is needed for static vs dynamic classification
    """
    # Define activity groups
    activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    static_activities = ['SITTING', 'STANDING', 'LAYING']
    dynamic_activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']
    
    combined_train_path = './Combined/Train'  # Adjust path as needed
    
    # Dictionary to store linear acceleration data
    linear_accel_data = {}
    activity_stats = {}
    
    # Load and calculate linear acceleration for each activity
    for activity in activities:
        activity_path = os.path.join(combined_train_path, activity)
        if os.path.exists(activity_path):
            files = [f for f in os.listdir(activity_path) if f.endswith('.csv')]
            all_linear_acc = []
            # Process each file for this activity
            for file in files[:5]:  # Use first 5 files to avoid memory issues
                file_path = os.path.join(activity_path, file)
                df = pd.read_csv(file_path)
                linear_acc_squared = df.iloc[:, 0]**2 + df.iloc[:, 1]**2 + df.iloc[:, 2]**2 # acc_x² + acc_y² + acc_z²
                all_linear_acc.extend(linear_acc_squared.values)
            
            linear_accel_data[activity] = np.array(all_linear_acc)
            
            # statistics
            activity_stats[activity] = {
                'mean': np.mean(all_linear_acc),
                'std': np.std(all_linear_acc),
                'median': np.median(all_linear_acc),
                'min': np.min(all_linear_acc),
                'max': np.max(all_linear_acc)
            }
            # print(f"Processed {activity}: {len(all_linear_acc)} data points")
    
    return linear_accel_data, activity_stats, static_activities, dynamic_activities

def create_linear_acceleration_visualizations(linear_accel_data, activity_stats, static_activities, dynamic_activities):
    """
    Create comprehensive visualizations for linear acceleration analysis
    """
    
    # 1. Box Plot Comparison
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Box plot of linear acceleration by activity
    plt.subplot(2, 2, 1)
    
    activities = list(linear_accel_data.keys())
    data_for_boxplot = [linear_accel_data[activity] for activity in activities]
    
    # Create box plot
    box_plot = plt.boxplot(data_for_boxplot, labels=activities, patch_artist=True)
    
    # Color code: blue for static, orange for dynamic
    colors = []
    for activity in activities:
        if activity in static_activities:
            colors.append('lightblue')
        else:
            colors.append('lightcoral')
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title('Linear Acceleration Distribution by Activity', fontsize=14, fontweight='bold')
    plt.xlabel('Activities')
    plt.ylabel('Linear Acceleration (g²)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Mean and Standard Deviation Bar Plot
    plt.subplot(2, 2, 2)
    
    means = [activity_stats[activity]['mean'] for activity in activities]
    stds = [activity_stats[activity]['std'] for activity in activities]
    
    x_pos = np.arange(len(activities))
    
    plt.bar(x_pos - 0.2, means, 0.4, label='Mean', color='skyblue', alpha=0.7)
    plt.bar(x_pos + 0.2, stds, 0.4, label='Std Dev', color='lightcoral', alpha=0.7)
    
    plt.xlabel('Activities')
    plt.ylabel('Linear Acceleration (g²)')
    plt.title('Mean and Standard Deviation Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, activities, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Static vs Dynamic Group Comparison
    plt.subplot(2, 2, 3)
    
    # Aggregate data by groups
    static_data = []
    dynamic_data = []
    
    for activity in static_activities:
        if activity in linear_accel_data:
            static_data.extend(linear_accel_data[activity])
    
    for activity in dynamic_activities:
        if activity in linear_accel_data:
            dynamic_data.extend(linear_accel_data[activity])
    
    # Create histograms
    plt.hist(static_data, bins=50, alpha=0.7, label='Static Activities', 
             color='lightblue', density=True)
    plt.hist(dynamic_data, bins=50, alpha=0.7, label='Dynamic Activities', 
             color='lightcoral', density=True)
    
    plt.xlabel('Linear Acceleration (g²)')
    plt.ylabel('Density')
    plt.title('Distribution Comparison: Static vs Dynamic', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Threshold Analysis
    plt.subplot(2, 2, 4)
    
    # Calculate group statistics
    static_mean = np.mean(static_data)
    dynamic_mean = np.mean(dynamic_data)
    static_std = np.std(static_data)
    dynamic_std = np.std(dynamic_data)
    
    # Proposed threshold
    threshold = (static_mean + dynamic_mean) / 2
    
    groups = ['Static\nActivities', 'Dynamic\nActivities']
    group_means = [static_mean, dynamic_mean]
    group_stds = [static_std, dynamic_std]
    
    plt.bar(groups, group_means, yerr=group_stds, capsize=5, 
            color=['lightblue', 'lightcoral'], alpha=0.7)
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Proposed Threshold: {threshold:.2f}')
    
    plt.ylabel('Linear Acceleration (g²)')
    plt.title('Group Means with Proposed Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('EDA-2-result.png', dpi=300, bbox_inches='tight')
    plt.show()
    return static_data, dynamic_data, threshold

def analyze_threshold_effectiveness(static_data, dynamic_data, threshold):
    """
    Calculates and prints the accuracy of a simple threshold classifier.
    """
    # Test threshold effectiveness
    static_correct = sum(np.array(static_data) < threshold)
    static_total = len(static_data)
    dynamic_correct = sum(np.array(dynamic_data) >= threshold)
    dynamic_total = len(dynamic_data)

    # Calculate overall accuracy
    overall_accuracy = (static_correct + dynamic_correct) / (static_total + dynamic_total) * 100

    print("\nTHRESHOLD ANALYSIS:")
    print(f"A simple threshold of {threshold:.2f} g² was used to separate static and dynamic activities.")
    print(f"This approach achieved an overall accuracy of {overall_accuracy:.1f}%.")
    return overall_accuracy


# ====================================================================================================================

# Step 1: Load and analyze data
linear_accel_data, activity_stats, static_activities, dynamic_activities = analyze_linear_acceleration_static_vs_dynamic()

# Step 2: Create visualizations
static_data, dynamic_data, threshold = create_linear_acceleration_visualizations(
    linear_accel_data, activity_stats, static_activities, dynamic_activities)

# Step 3: Simplified analysis and reporting
accuracy = analyze_threshold_effectiveness(static_data, dynamic_data, threshold)
