# **Task 1: Exploratory Data Analysis (EDA)**
This report details the exploratory data analysis conducted on the raw accelerometer data from the UCI-HAR dataset. The goal was to understand the underlying structure of the data, assess the feasibility of classification, and determine the necessity of feature engineering and machine learning models.

# 1\. Waveform Visualization and Initial Analysis

**Question:** Plot the waveform for one sample data from each activity class. Are you able to see any difference/similarities between the activities? Do you think the model will be able to classify the activities based on the data?
Methodology:
To get an initial understanding of the data, one sample from each of the six activity classes was visualized. The plots display the X, Y, and Z accelerometer readings over a 10-second window (500 samples).

Observations & Reasoning:

The waveforms, as seen in EDA-1-result.png, show a distinct difference between two primary categories of activities:

* **Dynamic Activities (WALKING, WALKING\_UPSTAIRS, WALKING\_DOWNSTAIRS):** These activities exhibit high-amplitude, high-frequency signals with no stable baseline. The acceleration values fluctuate rapidly and significantly, which is characteristic of continuous body movement.  
* **Static Activities (SITTING, STANDING, LAYING):** These activities show low-amplitude, low-frequency signals. The waveforms are much more stable and are primarily influenced by the constant force of gravity on the accelerometer, which results in consistent, non-zero readings on one or more axes depending on the phone's orientation.

Conclusion:
Yes, a model should be able to classify these activities. The visual difference between the static and dynamic groups is substantial. However, distinguishing between activities within the same group (e.g., SITTING vs. STANDING) appears more challenging from the raw waveform alone, as their signals are subtly different. This suggests that a machine learning model will be necessary to learn these nuanced patterns.

---

# 2\. Differentiating Static vs. Dynamic Activities

**Question:** Do you think we need a machine learning model to differentiate between static activities and dynamic activities? Look at the linear acceleration for each activity and justify your answer.

Methodology:

To answer this, I analyzed the magnitude of the linear acceleration, calculated as accx2​+accy2​+accz2​. The data was split into two groups: static and dynamic. A simple threshold-based classifier was designed to test if a basic rule could effectively separate the two groups. The performance of this simple rule was then evaluated.

Observations & Reasoning:

The analysis, visualized in EDA-2-result.png, revealed the following:

* The mean and standard deviation of linear acceleration are significantly different for the two groups. Dynamic activities have a higher mean and a vastly larger standard deviation, indicating more energy and variability.  
* The distribution plot (bottom left) shows that while the two groups have distinct central tendencies, there is a **significant region of overlap**.  
* A simple threshold classifier placed between the means of the two distributions achieved an accuracy of **76.9%**.

Conclusion:
Yes, a machine learning model is necessary even for this seemingly simple task. An accuracy of 76.9% demonstrates that a simple threshold is not robust enough for reliable classification, as it fails on nearly a quarter of the samples. The overlap in the distributions indicates that linear acceleration alone is not a perfect separator. A machine learning model would be able to learn a more complex decision boundary, potentially by incorporating other features, to accurately classify the ambiguous samples that fall within this overlapping region.

# 3\. PCA Visualization for Human Activity Recognition

## Objective

Visualize Human Activity Recognition data using Principal Component Analysis with three different approaches and compare their effectiveness for dimensionality reduction and activity discrimination.

## Methodology
### Method 1: PCA on Total Acceleration
- **Approach**: Calculate magnitude of 3D acceleration vectors using sqrt(acc_x² + acc_y² + acc_z²)
- **Features**: 500 features per sample (magnitude time series)
- **Rationale**: Direct physics-based approach using fundamental acceleration principles
- **Processing**: StandardScaler normalization followed by 2-component PCA

### Method 2: PCA on TSFEL Features
- **Approach**: Automated feature extraction using Time Series Feature Extraction Library
- **Implementation**: Used tsfel.get_features_by_domain() with comprehensive feature extraction
- **Features**: 468 features per sample covering statistical, temporal, and spectral domains
- **Sampling Rate**: 50Hz for proper frequency domain analysis
- **Error Handling**: Robust fallback mechanisms for failed extractions

### Method 3: PCA on Dataset-Style Features
- **Approach**: Manual UCI-HAR style feature engineering
- **Body Acceleration**: Gravity component removal through mean subtraction
- **Feature Types**:
  - Time domain features (mean, std) for original and body acceleration
  - Jerk features from first derivative of body acceleration
  - Magnitude features for total and body acceleration vectors
- **Features**: 28 carefully engineered features per sample
- **Focus**: Physically meaningful features for human movement analysis

## Results

### Explained Variance Performance

**Dataset Features: 66.4% explained variance**
- PC1: 0.498, PC2: 0.166
- Best overall performance for dimensionality reduction
- Highest efficiency with only 28 features
- Superior activity clustering and separation

**TSFEL Features: 59.1% explained variance**
- PC1: 0.483, PC2: 0.108
- Strong automated feature extraction performance
- Comprehensive 468-feature representation
- Good activity separation with some overlap

**Total Acceleration: 20.0% explained variance**
- PC1: 0.108, PC2: 0.093
- Baseline performance with limited discriminative power
- Simple magnitude calculation insufficient for complex activities
- Poor separation between similar activities

### Visualization Quality Assessment

**Dataset Features Visualization**:
- Excellent cluster separation between all activity classes
- Clear boundaries between static and dynamic activities
- Distinct patterns for walking variations visible
- Optimal for exploratory data analysis

**TSFEL Features Visualization**:
- Good activity clustering with moderate overlap
- Automated features capture relevant movement characteristics
- Reasonable separation despite higher dimensionality
- Comprehensive feature coverage shows promise

**Total Acceleration Visualization**:
- Limited separation capability between similar activities
- Basic static vs dynamic distinction partially visible
- Walking variants difficult to distinguish
- Insufficient for detailed activity analysis

## Analysis and Insights

### Why Dataset Features Excel

- **Domain-Specific Engineering**: Features designed specifically for human movement patterns
- **Gravity Compensation**: Explicit body acceleration separation removes noise
- **Movement Dynamics**: Jerk features capture acceleration changes crucial for activity discrimination
- **Feature Efficiency**: Achieves best results with minimal feature count, avoiding curse of dimensionality
- **Physical Meaning**: Each feature has clear interpretation in movement analysis context

### TSFEL Performance Analysis

**Strengths**:
- Automated extraction reduces manual engineering effort
- Comprehensive coverage across multiple feature domains
- Robust extraction with proper error handling
- Good performance without domain expertise

**Limitations**:
- Generic features may include irrelevant components for HAR
- High dimensionality (468 features) introduces potential noise
- Automated selection may miss domain-specific optimal features
- Computational overhead for feature extraction

### Total Acceleration Limitations

- **Information Loss**: Magnitude calculation discards directional movement information
- **Temporal Smoothing**: Individual axis dynamics lost in vector summation
- **Single Feature Type**: Insufficient diversity for complex activity patterns
- **High Dimensionality**: 500 timepoints with limited information density

## Key Findings

### Performance Hierarchy

- **Best Method**: Dataset Features achieve 66.4% explained variance with optimal efficiency
- **Second Best**: TSFEL Features provide 59.1% explained variance with automation benefits
- **Baseline**: Total Acceleration offers 20.0% explained variance as simple reference

### Feature Engineering Insights

- **Quality over Quantity**: 28 targeted features outperform 468 automated features
- **Domain Knowledge Critical**: Human movement understanding essential for optimal feature design
- **Gravity Separation Essential**: Body acceleration provides cleaner movement signals
- **Jerk Features Valuable**: Movement dynamics through derivatives highly discriminative

## Conclusions

### Primary Conclusions

- **Dataset-style feature engineering significantly outperforms automated extraction** for Human Activity Recognition PCA visualization
- **Feature quality is more important than feature quantity** in achieving effective dimensionality reduction
- **Domain knowledge integration is critical** for optimal HAR system performance
- **PCA visualization effectiveness depends heavily on feature choice** rather than algorithm parameters

### Recommendations

**For Future HAR Projects**:
- Implement domain-specific feature engineering as first priority
- Use TSFEL as automated alternative when domain expertise unavailable
- Always include gravity compensation and jerk features
- Validate feature effectiveness through dimensionality reduction analysis

Dataset Features method provides the optimal approach for PCA visualization in Human Activity Recognition, achieving superior explained variance (66.4%) with minimal computational overhead. The combination of domain-specific engineering, gravity compensation, and movement dynamics capture proves essential for effective HAR analysis and visualization.

***

# 4: Feature Correlation Analysis for Human Activity Recognition

## Objective

Analyze correlation patterns within TSFEL features and dataset-style features to identify redundancies, understand feature relationships, and provide recommendations for feature selection optimization in Human Activity Recognition systems.

## Methodology

### Data Preparation

- **Dataset**: 126 training samples from UCI HAR Combined dataset
- **Sample Structure**: 500 timepoints × 3 axes per sample
- **Activities**: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying
- **Analysis Focus**: Feature-to-feature correlation patterns

### Feature Extraction Methods

**TSFEL Features**:
- **Library**: Time Series Feature Extraction Library with comprehensive domain coverage
- **Extraction Process**: Automated feature generation using tsfel.get_features_by_domain()
- **Feature Count**: 468 features per sample
- **Domains**: Statistical, temporal, and spectral features across all axes
- **Sampling Rate**: 50Hz for proper frequency domain analysis

**Dataset-Style Features**:
- **Approach**: Manual UCI-HAR style feature engineering
- **Feature Categories**:
  - Body acceleration statistics (mean, std) after gravity removal
  - Gravity component statistics for each axis
  - Jerk features from acceleration derivatives
  - Magnitude features for total and body acceleration
- **Feature Count**: 22 features per sample
- **Design**: Physically meaningful features for human movement

### Correlation Analysis Process

**Correlation Calculation**:
- **Method**: Pearson correlation coefficient using np.corrcoef()
- **Matrix Generation**: Feature-to-feature correlation matrices
- **Range**: Correlation values from -1.0 to +1.0

**High Correlation Detection**:
- **Threshold**: |r| ≥ 0.8 for identifying strong linear relationships
- **Analysis**: Pairwise comparison of all feature combinations
- **Ranking**: Sorted by absolute correlation magnitude

**Redundancy Rate Calculation**:
- **Formula**: (High correlation pairs / Total possible pairs) × 100
- **Assessment**: Percentage of redundant feature relationships
- **Comparison**: Relative redundancy between feature sets

## Results

### Correlation Matrix Characteristics

**TSFEL Features Correlation Matrix**:
- **Dimensions**: 468 × 468 correlation matrix
- **Visual Patterns**: Clear block structures indicating feature clustering
- **Color Coding**: Red-blue diverging scheme showing correlation strength
- **Clustering**: Distinct groups of highly correlated features visible
- **Complexity**: High density matrix requiring index-based navigation

**Dataset Features Correlation Matrix**:
- **Dimensions**: 22 × 22 correlation matrix
- **Visual Patterns**: Sparse correlation structure with interpretable labels
- **Feature Names**: Readable labels showing specific feature relationships
- **Interpretability**: Clear mapping between feature names and correlation values
- **Structure**: More organized correlation patterns

### High Correlation Analysis

**TSFEL Features Redundancy**:
- **Pattern Observation**: Multiple blocks of highly correlated features
- **Statistical Clustering**: Features from same statistical domain show high correlation
- **Axis-Based Grouping**: Features from same acceleration axis cluster together
- **Frequency Domain**: Spectral features form distinct correlation groups
- **Cross-Domain**: Some correlations between statistical and frequency features

**Dataset Features Redundancy**:
- **Body-Gravity Relationships**: Expected correlations between related components
- **Magnitude Correlations**: Total and body magnitude features show relationships
- **Axis Consistency**: Similar patterns across X, Y, Z axes
- **Jerk Independence**: Jerk features show lower correlation with static features

### Redundancy Assessment

**TSFEL Features**:
- **Total Pairs**: 109,278 possible feature pairs
- **High Correlations**: Multiple significant correlation clusters observed
- **Redundancy Pattern**: Substantial redundancy indicated by block structure
- **Recommendation**: Aggressive feature selection required

**Dataset Features**:
- **Total Pairs**: 231 possible feature pairs
- **High Correlations**: Selective correlations with clear physical meaning
- **Redundancy Pattern**: Lower overall redundancy with interpretable relationships
- **Recommendation**: Minimal feature selection needed

## Analysis and Interpretation

### Correlation Patterns Identified

**TSFEL Feature Clustering**:
- **Within-Axis Clustering**: Features from same acceleration axis highly correlated
- **Statistical Group Correlations**: Mean, variance, standard deviation features cluster
- **Frequency Domain Blocks**: FFT-based features form distinct correlation groups
- **Cross-Domain Relationships**: Some statistical and spectral features correlated
- **Redundancy Implications**: Many features provide overlapping information

**Dataset Feature Relationships**:
- **Physical Correlations**: Body and gravity components show expected relationships
- **Magnitude Dependencies**: Total magnitude related to body magnitude as expected
- **Axis Symmetry**: Similar correlation patterns across X, Y, Z axes
- **Feature Independence**: Jerk features maintain independence from static measures
- **Interpretable Structure**: Correlations align with physical understanding

### Feature Redundancy Impact

**Computational Implications**:
- **TSFEL Processing**: 468 features require significant computational resources
- **Dataset Efficiency**: 22 features optimal for real-time applications
- **Memory Usage**: TSFEL features need ~20x more storage
- **Training Time**: More features increase model training duration

**Model Performance Effects**:
- **Overfitting Risk**: Redundant features increase overfitting likelihood
- **Noise Introduction**: Correlated features may amplify noise
- **Interpretability Loss**: Redundant features complicate model explanation
- **Generalization Impact**: High correlation may reduce model robustness

## Conclusions

### Primary Findings

- **TSFEL features exhibit significant redundancy** requiring aggressive feature selection for optimal performance
- **Dataset features show minimal redundancy** with interpretable correlation patterns aligned with physical principles
- **Correlation analysis reveals feature clustering** that guides effective feature selection strategies
- **Feature quality outweighs quantity** for Human Activity Recognition applications

### Key Insights

- **Automated feature extraction produces redundancy** that must be addressed through selection techniques
- **Domain-specific engineering minimizes redundancy** while maintaining discriminative power
- **Correlation patterns provide actionable insights** for feature optimization
- **Computational efficiency requires redundancy management** especially for TSFEL features

### Recommendations Summary

**For TSFEL Usage**:
- **Essential**: Implement correlation-based feature selection
- **Target**: Reduce to 50-100 most informative features
- **Method**: Combine correlation filtering with domain expertise
- **Validation**: Cross-validate feature subsets for performance

**For Dataset Features**:
- **Approach**: Minimal feature selection required
- **Maintenance**: Monitor correlation patterns over time
- **Enhancement**: Add features only with demonstrated performance gains
- **Priority**: Maintain interpretability and physical meaning

Correlation analysis confirms that domain-specific feature engineering (dataset features) provides superior efficiency with minimal redundancy, while automated extraction (TSFEL features) requires careful redundancy management for optimal performance. Feature selection is critical for TSFEL usage but minimal for dataset features.

This exploratory data analysis revealed several key insights. While raw accelerometer waveforms show a clear visual distinction between static and dynamic activities, a simple threshold on linear acceleration is insufficient for reliable classification, justifying the need for machine learning. Furthermore, the effectiveness of visualization and dimensionality reduction via PCA is heavily dependent on robust feature engineering, with statistical features providing the clearest class separation. Finally, the analysis confirmed the presence of redundant features, indicating that feature selection will be a crucial step in building an efficient final model.

