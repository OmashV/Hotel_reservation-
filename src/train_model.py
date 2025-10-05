import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings
import time
import os
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("HOTEL RESERVATION PREDICTION")
print("="*60)

# 1. LOAD DATA
# ------------
data_path = "data/hotel_reservations.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Please place 'hotel_reservations.csv' in the 'data' folder.")
df = pd.read_csv(data_path)

print(f"\n✓ Data loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 2. DATA INSPECTION
# ------------------
print("\n" + "="*60)
print("DATA INSPECTION")
print("="*60)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data Info ---")
print(df.info())

print("\n--- Missing Values ---")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values!")

print("\n--- Target Variable Distribution ---")
print(df['booking_status'].value_counts())
print("\nPercentages:")
print(df['booking_status'].value_counts(normalize=True) * 100)

# 3. EXPLORATORY DATA ANALYSIS
# -----------------------------
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# Summary statistics
print("\n--- Numerical Features Summary ---")
print(df.describe())

# Target distribution visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes = axes.ravel()

df['booking_status'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'], edgecolor='black')
axes[0].set_title('Booking Status Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Status', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)

df['booking_status'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                         colors=['green', 'red'], startangle=90)
axes[1].set_title('Booking Status Percentage', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig("models/target_distribution.png")
plt.close()

# Key numerical features distribution
numerical_features = ['lead_time', 'avg_price_per_room', 'no_of_adults', 
                     'no_of_weekend_nights', 'no_of_week_nights', 'no_of_special_requests']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, feature in enumerate(numerical_features):
    if feature in df.columns:
        axes[idx].hist(df[feature].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {feature}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("models/numerical_distribution.png")
plt.close()

# Categorical features distribution
categorical_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes = axes.ravel()

for idx, feature in enumerate(categorical_features):
    if feature in df.columns:
        df[feature].value_counts().plot(kind='bar', ax=axes[idx], color='coral', edgecolor='black')
        axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(feature, fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("models/categorical_distribution.png")
plt.close()

# 4. DATA PREPROCESSING
# ----------------------
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Create working copy
df_work = df.copy()

# Step 1: Drop Booking_ID, arrival_year, and low-importance features
print("\n--- Step 1: Remove ID and irrelevant columns ---")
columns_to_drop = ['Booking_ID', 'arrival_year', 'arrival_date', 
                   'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled']
df_work = df_work.drop(columns=[col for col in columns_to_drop if col in df_work.columns], axis=1)
print(f"✓ Dropped columns: {[col for col in columns_to_drop if col in df_work.columns or col in columns_to_drop]}")

# Step 2: Handle missing values
print("\n--- Step 2: Handle Missing Values ---")
missing_before = df_work.isnull().sum().sum()
print(f"Total missing values before: {missing_before}")

for col in df_work.columns:
    if df_work[col].isnull().sum() > 0:
        missing_count = df_work[col].isnull().sum()
        missing_pct = (missing_count / len(df_work)) * 100
        
        if df_work[col].dtype == 'object':
            mode_val = df_work[col].mode()[0]
            df_work[col].fillna(mode_val, inplace=True)
            print(f"  {col}: Filled {missing_count} ({missing_pct:.2f}%) missing with mode '{mode_val}'")
        else:
            median_val = df_work[col].median()
            df_work[col].fillna(median_val, inplace=True)
            print(f"  {col}: Filled {missing_count} ({missing_pct:.2f}%) missing with median {median_val:.2f}")

if df_work.isnull().sum().sum() == 0:
    print("✓ No missing values remaining")

# Step 3: Remove duplicates
print("\n--- Step 3: Remove Duplicate Rows ---")
duplicates = df_work.duplicated().sum()
if duplicates > 0:
    print(f"Found {duplicates} duplicate rows ({duplicates/len(df_work)*100:.2f}%)")
    df_work = df_work.drop_duplicates()
    print(f"✓ Removed duplicates. New shape: {df_work.shape}")
else:
    print("✓ No duplicate rows found")

# Step 4: Handle outliers in numerical features
print("\n--- Step 4: Handle Outliers (IQR Method) ---")
numerical_features_pre = df_work.select_dtypes(include=[np.number]).columns.tolist()
if 'booking_status' in numerical_features_pre:
    numerical_features_pre.remove('booking_status')

outliers_summary = []
for col in numerical_features_pre:
    Q1 = df_work[col].quantile(0.25)
    Q3 = df_work[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers_count = ((df_work[col] < lower_bound) | (df_work[col] > upper_bound)).sum()
    
    if outliers_count > 0:
        outliers_pct = (outliers_count / len(df_work)) * 100
        df_work[col] = df_work[col].clip(lower_bound, upper_bound)
        outliers_summary.append({
            'Feature': col,
            'Outliers': outliers_count,
            'Percentage': f"{outliers_pct:.2f}%"
        })
        print(f"  {col}: Capped {outliers_count} outliers ({outliers_pct:.2f}%)")

if outliers_summary:
    print(f"✓ Handled outliers in {len(outliers_summary)} features")
else:
    print("✓ No significant outliers detected")

# Step 5: Feature Engineering
print("\n--- Step 5: Feature Engineering ---")

# Total number of guests
if 'no_of_adults' in df_work.columns and 'no_of_children' in df_work.columns:
    df_work['total_guests'] = df_work['no_of_adults'] + df_work['no_of_children']
    print("✓ Created 'total_guests' feature")

# Total nights
if 'no_of_weekend_nights' in df_work.columns and 'no_of_week_nights' in df_work.columns:
    df_work['total_nights'] = df_work['no_of_weekend_nights'] + df_work['no_of_week_nights']
    print("✓ Created 'total_nights' feature")

# Price per night
if 'avg_price_per_room' in df_work.columns and 'total_nights' in df_work.columns:
    df_work['price_per_night'] = df_work['avg_price_per_room'] / (df_work['total_nights'] + 1)
    print("✓ Created 'price_per_night' feature")

# Is weekend booking
if 'no_of_weekend_nights' in df_work.columns:
    df_work['is_weekend_booking'] = (df_work['no_of_weekend_nights'] > 0).astype(int)
    print("✓ Created 'is_weekend_booking' feature")

# Lead time categories
if 'lead_time' in df_work.columns:
    df_work['lead_time_category'] = pd.cut(df_work['lead_time'], 
                                           bins=[0, 7, 30, 90, 365, 1000],
                                           labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long'])
    print("✓ Created 'lead_time_category' feature")

# Guest loyalty score
if 'repeated_guest' in df_work.columns:
    df_work['loyalty_score'] = df_work['repeated_guest']
    print("✓ Created 'loyalty_score' feature")

# Has special requests
if 'no_of_special_requests' in df_work.columns:
    df_work['has_special_requests'] = (df_work['no_of_special_requests'] > 0).astype(int)
    print("✓ Created 'has_special_requests' feature")

# Interaction term: lead_time * avg_price_per_room
if 'lead_time' in df_work.columns and 'avg_price_per_room' in df_work.columns:
    df_work['lead_time_price_interaction'] = df_work['lead_time'] * df_work['avg_price_per_room']
    print("✓ Created 'lead_time_price_interaction' feature")

# Interaction term: market_segment_type * no_of_special_requests
if 'market_segment_type' in df_work.columns and 'no_of_special_requests' in df_work.columns:
    df_work['market_segment_special_requests'] = df_work['no_of_special_requests']
    print("✓ Created 'market_segment_special_requests' feature (placeholder, encoded later)")

print(f"\n✓ Total features after engineering: {df_work.shape[1] - 1}")

# Step 6: Encode target variable
print("\n--- Step 6: Encode Target Variable ---")
target_mapping = {'Not_Canceled': 0, 'Canceled': 1}
df_work['booking_status'] = df_work['booking_status'].map(target_mapping)
print(f"Target encoding: {target_mapping}")
print(f"Encoded distribution:\n{df_work['booking_status'].value_counts()}")

# Verify encoding
if df_work['booking_status'].isnull().any():
    print("❌ ERROR: Target encoding failed!")
    raise ValueError("Check target column values")

# Step 7: Separate features and target
print("\n--- Step 7: Separate Features and Target ---")
y = df_work['booking_status'].values
X = df_work.drop('booking_status', axis=1)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Features: {list(X.columns)}")

# Step 8: Identify categorical and numerical features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n--- Step 8: Feature Types ---")
print(f"Categorical ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical ({len(numerical_cols)}): {numerical_cols}")

# 5. SPLIT DATA
# -------------
print("\n--- Step 9: Train-Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Train target distribution: {np.bincount(y_train)}")
print(f"Test target distribution: {np.bincount(y_test)}")

# 6. ENCODE CATEGORICAL FEATURES
# -------------------------------
print("\n--- Step 10: Encode Categorical Features ---")
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    
    # Fit on train
    X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
    
    # Handle unseen categories in test
    test_values = X_test[col].astype(str)
    unseen = set(test_values.unique()) - set(le.classes_)
    
    if unseen:
        print(f"  {col}: {len(unseen)} unseen categories in test, mapping to most frequent")
        most_frequent = X_train[col].mode()[0]
        test_values = test_values.apply(lambda x: most_frequent if x in unseen else x)
    
    X_test_encoded[col] = le.transform(test_values)
    label_encoders[col] = le
    
    print(f"  ✓ {col} encoded ({len(le.classes_)} unique values)")

# Create interaction term for market_segment_type * no_of_special_requests
if 'market_segment_type' in X_train_encoded.columns and 'no_of_special_requests' in X_train_encoded.columns:
    X_train_encoded['market_segment_special_requests'] = X_train_encoded['market_segment_type'] * X_train_encoded['no_of_special_requests']
    X_test_encoded['market_segment_special_requests'] = X_test_encoded['market_segment_type'] * X_test_encoded['no_of_special_requests']
    numerical_cols.append('market_segment_special_requests')
    print("✓ Created encoded 'market_segment_special_requests' feature")

# 7. POLYNOMIAL FEATURES
# ----------------------
print("\n--- Step 11: Add Polynomial Features ---")
poly_features = ['lead_time', 'avg_price_per_room']
if all(f in X_train_encoded.columns for f in poly_features):
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_cols = [f for f in poly_features if f in X_train_encoded.columns]
    X_train_poly = pd.DataFrame(
        poly.fit_transform(X_train_encoded[poly_cols]),
        columns=poly.get_feature_names_out(poly_cols),
        index=X_train_encoded.index
    )
    X_test_poly = pd.DataFrame(
        poly.transform(X_test_encoded[poly_cols]),
        columns=poly.get_feature_names_out(poly_cols),
        index=X_test_encoded.index
    )
    
    # Drop original poly features and add new ones
    X_train_encoded = X_train_encoded.drop(columns=poly_cols)
    X_test_encoded = X_test_encoded.drop(columns=poly_cols)
    X_train_encoded = pd.concat([X_train_encoded, X_train_poly], axis=1)
    X_test_encoded = pd.concat([X_test_encoded, X_test_poly], axis=1)
    
    numerical_cols = [col for col in numerical_cols if col not in poly_cols] + list(poly.get_feature_names_out(poly_cols))
    print(f"✓ Added polynomial features: {list(poly.get_feature_names_out(poly_cols))}")

# Print feature order for debugging
print("\n--- Feature Order for Scaling ---")
print(f"Features: {list(X_train_encoded.columns)}")

# 8. FEATURE SCALING
# ------------------
print("\n--- Step 12: Feature Scaling ---")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# Handle any NaN/Inf values
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

print(f"✓ Features scaled")
print(f"Train scaled shape: {X_train_scaled.shape}")
print(f"Test scaled shape: {X_test_scaled.shape}")
print(f"Train data range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
print(f"Test data range: [{X_test_scaled.min():.2f}, {X_test_scaled.max():.2f}]")

# 9. FEATURE SELECTION
# --------------------
print("\n" + "="*60)
print("FEATURE SELECTION")
print("="*60)

# Train a preliminary RandomForest to get feature importances
prelim_model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
prelim_model.fit(X_train_scaled, y_train)

feature_importance = pd.DataFrame({
    'Feature': X_train_encoded.columns,
    'Importance': prelim_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nPreliminary Feature Importance:")
print(feature_importance.to_string(index=False))

# Drop features with importance < 0.005
low_importance_features = feature_importance[feature_importance['Importance'] < 0.005]['Feature'].tolist()
if low_importance_features:
    print(f"\nDropping low-importance features: {low_importance_features}")
    X_train_encoded = X_train_encoded.drop(columns=low_importance_features)
    X_test_encoded = X_test_encoded.drop(columns=low_importance_features)
    
    # Re-scale after dropping features
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"✓ New feature set: {list(X_train_encoded.columns)}")
else:
    print("✓ No low-importance features to drop")

# 10. APPLY SMOTE
# ---------------
print("\n" + "="*60)
print("APPLY SMOTE")
print("="*60)

smote = SMOTE(sampling_strategy=0.95, k_neighbors=3, random_state=42)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
print(f"New training set shape: {X_train_scaled.shape}")
print(f"New target distribution: {np.bincount(y_train)}")

# 11. BASELINE MODELING
# ---------------------
print("\n" + "="*60)
print("BASELINE MODELING")
print("="*60)

models = {
    'RandomForestClassifier': RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'ExtraTreesClassifier': ExtraTreesClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'DecisionTreeClassifier': DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    ),
    'StackingClassifier': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=3,
        n_jobs=-1
    )
}

baseline_results = []

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    train_roc = roc_auc_score(y_train, y_train_prob)
    test_roc = roc_auc_score(y_test, y_test_prob)
    
    # Per-class metrics
    class_report = classification_report(y_train, y_train_pred, output_dict=True)
    test_class_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    print(f"  Train Accuracy: {train_acc:.6f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy:  {test_acc:.6f} ({test_acc*100:.2f}%)")
    print(f"  Train F1:       {train_f1:.6f}")
    print(f"  Test F1:        {test_f1:.6f}")
    print(f"  Train ROC AUC:  {train_roc:.6f}")
    print(f"  Test ROC AUC:   {test_roc:.6f}")
    print(f"  Test Recall (Canceled): {test_class_report['1']['recall']:.6f}")
    
    baseline_results.append({
        'Model': name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Train F1': train_f1,
        'Test F1': test_f1,
        'Train ROC AUC': train_roc,
        'Test ROC AUC': test_roc,
        'Test Recall (Canceled)': test_class_report['1']['recall']
    })

baseline_df = pd.DataFrame(baseline_results)
print("\n" + "="*60)
print("BASELINE RESULTS SUMMARY")
print("="*60)
print(baseline_df.to_string(index=False))

# 12. HYPERPARAMETER TUNING
# -------------------------
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)

param_grids = {
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'class_weight': ['balanced']
    },
    'ExtraTreesClassifier': {
        'n_estimators': [50, 100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'class_weight': ['balanced']
    },
    'DecisionTreeClassifier': {
        'max_depth': [8, 12, 16, 20, 24, None],
        'min_samples_split': [5, 10, 15, 20, 25],
        'min_samples_leaf': [2, 5, 10, 15, 20],
        'class_weight': ['balanced']
    },
    'StackingClassifier': {
        'rf__n_estimators': [50, 100, 200, 300],
        'rf__max_depth': [10, 20, 30, 40, None],
        'et__n_estimators': [50, 100, 200, 300],
        'et__max_depth': [10, 20, 30, 40, None]
    }
}

tuned_results = []
best_models = {}

for name in models.keys():
    print(f"\n--- Tuning {name} ---")
    print("This may take a few minutes...")
    
    start_time = time.time()
    
    grid_search = RandomizedSearchCV(
        models[name].__class__() if name not in ['StackingClassifier'] else models[name],
        param_grids[name],
        n_iter=30,  # Increased for better tuning
        cv=3,
        scoring='recall',  # Optimize for recall of "Canceled"
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    time_taken = time.time() - start_time
    
    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV Recall: {grid_search.best_score_:.6f}")
    
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    # Evaluate best model
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    y_train_prob = best_model.predict_proba(X_train_scaled)[:, 1]
    y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    train_roc = roc_auc_score(y_train, y_train_prob)
    test_roc = roc_auc_score(y_test, y_test_prob)
    test_class_report = classification_report(y_test, y_test_pred, output_dict=True)
    
    print(f"  Train Accuracy: {train_acc:.6f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy:  {test_acc:.6f} ({test_acc*100:.2f}%)")
    print(f"  Test Recall (Canceled): {test_class_report['1']['recall']:.6f}")
    
    tuned_results.append({
        'Model': name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Train F1': train_f1,
        'Test F1': test_f1,
        'Train ROC AUC': train_roc,
        'Test ROC AUC': test_roc,
        'Test Recall (Canceled)': test_class_report['1']['recall'],
        'Time taken for tuning (s)': time_taken
    })

tuned_df = pd.DataFrame(tuned_results)
print("\n" + "="*60)
print("TUNING RESULTS SUMMARY")
print("="*60)
print(tuned_df.to_string(index=False))

# 13. EVALUATION
# --------------
print("\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

# Select best model based on weighted score (0.6 * Recall + 0.4 * Accuracy)
tuned_df['Weighted Score'] = 0.6 * tuned_df['Test Recall (Canceled)'] + 0.4 * tuned_df['Test Accuracy']
best_idx = tuned_df['Weighted Score'].idxmax()
best_model_name = tuned_df.loc[best_idx, 'Model']
best_model = best_models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"✓ Test Accuracy: {tuned_df.loc[best_idx, 'Test Accuracy']:.6f} ({tuned_df.loc[best_idx, 'Test Accuracy']*100:.2f}%)")
print(f"✓ Test F1 Score: {tuned_df.loc[best_idx, 'Test F1']:.6f}")
print(f"✓ Test ROC AUC: {tuned_df.loc[best_idx, 'Test ROC AUC']:.6f}")
print(f"✓ Test Recall (Canceled): {tuned_df.loc[best_idx, 'Test Recall (Canceled)']:.6f}")

# Save model and preprocessing objects
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")
joblib.dump(poly, "models/poly.pkl")
print("\n✓ Saved model and preprocessing objects to 'models' directory")

# Get predictions
y_test_pred = best_model.predict(X_test_scaled)

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_test_pred, 
                           target_names=['Not Canceled', 'Canceled']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Canceled', 'Canceled'],
            yticklabels=['Not Canceled', 'Canceled'],
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.close()

# Feature Importance
if hasattr(best_model, 'feature_importances_'):
    print("\n--- Feature Importance Analysis ---")
    
    feature_importance = pd.DataFrame({
        'Feature': X_train_encoded.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top 15 Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("models/feature_importance.png")
    plt.close()

# Model Comparison Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.ravel()

metrics = ['Test Accuracy', 'Test F1', 'Test Recall (Canceled)']
colors = ['steelblue', 'coral', 'lightgreen']

for idx, metric in enumerate(metrics):
    tuned_df.plot(x='Model', y=metric, kind='bar', ax=axes[idx], 
                  color=colors[idx], legend=False, edgecolor='black')
    axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Model', fontsize=12)
    axes[idx].set_ylabel(metric, fontsize=12)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].axhline(y=0.80, color='red', linestyle='--', linewidth=2, label='80% Target')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("models/model_comparison.png")
plt.close()

# Final Summary
print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)

print(f"\n📊 Final Results:")
print(f"  • Best Model: {best_model_name}")
print(f"  • Test Accuracy: {tuned_df.loc[best_idx, 'Test Accuracy']*100:.2f}%")
print(f"  • Test F1 Score: {tuned_df.loc[best_idx, 'Test F1']:.4f}")
print(f"  • Test ROC AUC: {tuned_df.loc[best_idx, 'Test ROC AUC']:.4f}")
print(f"  • Test Recall (Canceled): {tuned_df.loc[best_idx, 'Test Recall (Canceled)']:.4f}")

if tuned_df.loc[best_idx, 'Test Accuracy'] >= 0.85 and tuned_df.loc[best_idx, 'Test Recall (Canceled)'] >= 0.80:
    print("\n🎉 SUCCESS! Achieved target accuracy of 85%+ and recall of 80%+ for Canceled")
    print("Model is ready for deployment!")
else:
    print(f"\n⚠️ Current accuracy: {tuned_df.loc[best_idx, 'Test Accuracy']*100:.2f}%")
    print(f"⚠️ Current recall (Canceled): {tuned_df.loc[best_idx, 'Test Recall (Canceled)']*100:.2f}%")
    print("Consider adjusting decision threshold in app.py or further feature engineering.")

print("\n✓ All done! 🚀")