import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_activity_waveforms():
    """
    Load one sample from each activity class and plot accelerometer waveforms
    """
    activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    combined_train_path = './Combined/Train'     
    
    # Dictionary to store sample data for each activity
    sample_data = {}
    
    # Load one sample file from each activity
    for activity in activities:
        activity_path = os.path.join(combined_train_path, activity)
        
        if os.path.exists(activity_path):
            files = [f for f in os.listdir(activity_path) if f.endswith('.csv')]
            
            if files:
                sample_file = files[0]
                file_path = os.path.join(activity_path, sample_file)
                df = pd.read_csv(file_path)
                sample_data[activity] = df
                print(f"Loaded {activity}: {len(df)} data points")
    
    # PLOTS
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    colors = ["#FA5A5A", "#EADD22", "#93D8A0"]
    axis_labels = ['X-axis', 'Y-axis', 'Z-axis']
    
    for idx, activity in enumerate(activities):
        if activity in sample_data:
            data = sample_data[activity]
            time_points = len(data)
            time_axis = np.arange(time_points) / 50.0
            
            # Plot X, Y, Z acceleration components
            axes[idx].plot(time_axis, data.iloc[:, 0], color=colors[0], 
                          label=axis_labels[0], linewidth=1.2, alpha=0.8)
            axes[idx].plot(time_axis, data.iloc[:, 1], color=colors[1], 
                          label=axis_labels[1], linewidth=1.2, alpha=0.8)
            axes[idx].plot(time_axis, data.iloc[:, 2], color=colors[2], 
                          label=axis_labels[2], linewidth=1.2, alpha=0.8)
            
            axes[idx].set_title(f'{activity}', fontsize=12, fontweight='bold', pad=15)
            axes[idx].set_xlabel('Time (seconds)', fontsize=10)
            axes[idx].set_ylabel('Acceleration (g)', fontsize=10)
            
            axes[idx].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), 
                           ncol=3, fontsize=9, frameon=False)
            
            axes[idx].grid(True, alpha=0.3, linewidth=0.5)
            axes[idx].set_xlim(0, time_points/50.0)
            
            axes[idx].tick_params(axis='both', which='major', labelsize=9)
    
    fig.suptitle('Accelerometer Waveforms for Human Activity Recognition\n' + 
                 'Comparison of Static vs Dynamic Activities', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(
        top=0.88,        # Space for main title
        bottom=0.15,     # Space for legends below
        hspace=0.4,      # Vertical space between subplots
        wspace=0.3       # Horizontal space between subplots
    )
    plt.savefig('EDA-1-result.png', dpi=300, bbox_inches='tight')
    plt.show()
    return sample_data


# ================================================================================
sample_data = load_and_plot_activity_waveforms()
