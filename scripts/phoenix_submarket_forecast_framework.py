#!/usr/bin/env python3
"""
Phoenix Submarket Rent Growth Forecast Framework
Production-Validated Methodology for Submarket Analysis

Demonstrates how to apply the production-validated ensemble approach
to submarket-level forecasting in Phoenix MSA.

Two approaches supported:
1. Separate models per submarket (recommended if ≥50 quarters data)
2. Unified model with submarket features (better for <50 quarters)

Created: November 8, 2025
Author: Based on COMPLETE_ROOT_CAUSE_ANALYSIS.md findings
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# ============================================================================
# PRODUCTION CONFIGURATION (Validated from Root Cause Analysis)
# ============================================================================

PRODUCTION_CONFIG = {
    'sarima': {
        'order': (1, 1, 2),
        'seasonal_order': (0, 0, 1, 4),
    },
    'lightgbm': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 6,
        'min_data_in_leaf': 10,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1
    },
    'ridge': {
        'alphas': [0.1, 1.0, 10.0, 100.0, 1000.0],
        'cv': 5,
    },
    'early_stopping': 50,
    'validation_thresholds': {
        'sarima_max_prediction': 10.0,
        'component_correlation_min': -0.5,
        'min_data_quarters': 40,  # Minimum for separate models
    }
}

# Phoenix submarkets (example - update with actual submarkets)
PHOENIX_SUBMARKETS = [
    'Downtown Phoenix',
    'North Phoenix',
    'Scottsdale',
    'Tempe',
    'Mesa',
    'Chandler',
    'Glendale',
    'Paradise Valley',
    'Ahwatukee',
    'Arcadia'
]

# ============================================================================
# DATA GENERATION (Template for when real data becomes available)
# ============================================================================

def generate_template_submarket_data():
    """
    Generate template submarket data structure.

    REPLACE THIS FUNCTION with actual submarket data loading when available.

    Expected data structure:
    - date: Quarterly dates
    - submarket: Submarket name
    - rent_growth_yoy: Target variable
    - asking_rent: Average asking rent
    - vacancy_rate: Submarket vacancy
    - inventory_units: Total units in submarket
    - absorption: Net absorption
    - [All 25 core features from MSA-level model]
    """

    print("="*80)
    print("TEMPLATE SUBMARKET DATA GENERATION")
    print("="*80)
    print("⚠️  NOTE: This generates synthetic data for demonstration.")
    print("   Replace with actual submarket data from CoStar/Axiometrics/REIS")
    print()

    # Load MSA-level data as base
    msa_data = pd.read_csv('data/processed/phoenix_modeling_dataset.csv',
                          parse_dates=['date'])

    # Generate realistic submarket variations
    submarkets_data = []

    for submarket in PHOENIX_SUBMARKETS:
        sub_df = msa_data.copy()
        sub_df['submarket'] = submarket

        # Create realistic variations by submarket
        # (In real implementation, this comes from actual data)
        np.random.seed(hash(submarket) % 10000)

        # Vary rent growth by submarket characteristics
        if submarket in ['Downtown Phoenix', 'Tempe', 'Scottsdale']:
            # Urban core: Higher volatility
            sub_df['rent_growth_yoy'] += np.random.normal(0.5, 1.5, len(sub_df))
        elif submarket in ['Paradise Valley', 'Arcadia']:
            # Luxury: More stable, premium
            sub_df['rent_growth_yoy'] += np.random.normal(0.8, 0.8, len(sub_df))
        else:
            # Suburban: Moderate
            sub_df['rent_growth_yoy'] += np.random.normal(0, 1.0, len(sub_df))

        # Vary vacancy rates
        sub_df['vacancy_rate'] += np.random.normal(0, 1.0, len(sub_df))
        sub_df['vacancy_rate'] = sub_df['vacancy_rate'].clip(0, 20)

        # Vary asking rents
        rent_multiplier = {
            'Paradise Valley': 1.4,
            'Scottsdale': 1.3,
            'Arcadia': 1.25,
            'Downtown Phoenix': 1.15,
            'Tempe': 1.1,
            'Chandler': 1.05,
            'North Phoenix': 1.0,
            'Mesa': 0.95,
            'Glendale': 0.9,
            'Ahwatukee': 1.05
        }.get(submarket, 1.0)

        sub_df['asking_rent'] *= rent_multiplier

        submarkets_data.append(sub_df)

    # Combine all submarkets
    full_data = pd.concat(submarkets_data, ignore_index=True)
    full_data = full_data.sort_values(['submarket', 'date']).reset_index(drop=True)

    print(f"✅ Generated data for {len(PHOENIX_SUBMARKETS)} submarkets")
    print(f"   Total records: {len(full_data):,}")
    print(f"   Date range: {full_data['date'].min()} to {full_data['date'].max()}")

    return full_data

def save_template_data(data, output_path='data/processed/phoenix_submarket_template.csv'):
    """Save template submarket data"""
    data.to_csv(output_path, index=False)
    print(f"\n✅ Template data saved: {output_path}")
    print(f"   Update this file with actual submarket data when available")

# ============================================================================
# APPROACH 1: SEPARATE MODELS PER SUBMARKET
# ============================================================================

class SubmarketEnsembleModel:
    """
    Separate ensemble model for each submarket.

    Use when: Each submarket has ≥40 quarters of data
    Benefits: Captures submarket-specific dynamics
    Drawbacks: Requires more data per submarket
    """

    def __init__(self, submarket_name: str, config: dict = PRODUCTION_CONFIG):
        self.submarket = submarket_name
        self.config = config
        self.lgb_model = None
        self.sarima_model = None
        self.ridge_model = None
        self.scaler = None
        self.features = None

    def train(self, X_train, y_train, features):
        """Train submarket-specific ensemble"""
        print(f"\n{'='*80}")
        print(f"Training Ensemble for: {self.submarket}")
        print(f"{'='*80}")

        self.features = features

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Split for validation
        n_train = int(len(X_scaled) * 0.8)
        X_tr, X_val = X_scaled[:n_train], X_scaled[n_train:]
        y_tr, y_val = y_train[:n_train], y_train[n_train:]

        # Train LightGBM
        print(f"\n[1/3] Training LightGBM...")
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.lgb_model = lgb.train(
            self.config['lightgbm'],
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.config['early_stopping']),
                lgb.log_evaluation(period=0)
            ]
        )

        # Train SARIMA
        print(f"[2/3] Training SARIMA...")
        self.sarima_model = SARIMAX(
            y_train,
            order=self.config['sarima']['order'],
            seasonal_order=self.config['sarima']['seasonal_order'],
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        # Train Ridge meta-learner
        print(f"[3/3] Training Ridge Meta-Learner...")
        lgb_pred = self.lgb_model.predict(X_scaled)
        sarima_pred = self.sarima_model.fittedvalues

        X_meta = np.column_stack([lgb_pred, sarima_pred])

        self.ridge_model = RidgeCV(
            alphas=self.config['ridge']['alphas'],
            cv=TimeSeriesSplit(n_splits=self.config['ridge']['cv'])
        ).fit(X_meta, y_train)

        print(f"\n✅ {self.submarket} model trained")
        print(f"   Ridge alpha: {self.ridge_model.alpha_:.4f}")

    def predict(self, X_future) -> np.ndarray:
        """Generate predictions for submarket"""
        X_scaled = self.scaler.transform(X_future)

        # Component predictions
        lgb_pred = self.lgb_model.predict(X_scaled)
        sarima_pred = self.sarima_model.forecast(steps=len(X_future))

        # Ensemble
        X_meta = np.column_stack([lgb_pred, sarima_pred])
        ensemble_pred = self.ridge_model.predict(X_meta)

        return ensemble_pred

    def save(self, output_dir='models/submarkets'):
        """Save submarket model"""
        os.makedirs(output_dir, exist_ok=True)

        model_name = self.submarket.lower().replace(' ', '_')

        joblib.dump(self.lgb_model, f'{output_dir}/{model_name}_lgb.pkl')
        joblib.dump(self.sarima_model, f'{output_dir}/{model_name}_sarima.pkl')
        joblib.dump(self.ridge_model, f'{output_dir}/{model_name}_ridge.pkl')
        joblib.dump(self.scaler, f'{output_dir}/{model_name}_scaler.pkl')

        print(f"✅ {self.submarket} models saved to {output_dir}/")

def train_separate_submarket_models(data, features, test_cutoff='2022-12-31'):
    """
    Train separate ensemble model for each submarket.

    Recommended when: Each submarket has ≥40 quarters of data
    """
    print("\n" + "="*80)
    print("APPROACH 1: SEPARATE MODELS PER SUBMARKET")
    print("="*80)

    submarket_models = {}

    for submarket in PHOENIX_SUBMARKETS:
        print(f"\n{'='*80}")
        print(f"Processing: {submarket}")
        print(f"{'='*80}")

        # Filter data for this submarket
        sub_data = data[data['submarket'] == submarket].copy()

        # Check minimum data requirement
        if len(sub_data) < PRODUCTION_CONFIG['validation_thresholds']['min_data_quarters']:
            print(f"⚠️  WARNING: {submarket} has only {len(sub_data)} quarters")
            print(f"   Minimum recommended: {PRODUCTION_CONFIG['validation_thresholds']['min_data_quarters']}")
            print(f"   Consider using unified model approach instead")
            continue

        # Train/test split
        train = sub_data[sub_data['date'] <= test_cutoff]

        if len(train) < 30:
            print(f"⚠️  SKIP: {submarket} insufficient training data ({len(train)} quarters)")
            continue

        # Prepare features
        X_train = train[features].fillna(method='ffill')
        y_train = train['rent_growth_yoy']

        # Train model
        model = SubmarketEnsembleModel(submarket)
        model.train(X_train, y_train, features)
        model.save()

        submarket_models[submarket] = model

    print(f"\n{'='*80}")
    print(f"✅ Trained {len(submarket_models)} submarket models")
    print(f"{'='*80}")

    return submarket_models

# ============================================================================
# APPROACH 2: UNIFIED MODEL WITH SUBMARKET FEATURES
# ============================================================================

def create_submarket_features(data):
    """
    Create one-hot encoded submarket features for unified model.

    Use when: Submarkets have <40 quarters of individual data
    Benefits: Shares information across submarkets
    Drawbacks: May miss submarket-specific dynamics
    """
    print("\n" + "="*80)
    print("Creating Submarket Features for Unified Model")
    print("="*80)

    # One-hot encode submarkets
    submarket_dummies = pd.get_dummies(data['submarket'], prefix='subm')

    # Drop one category to avoid multicollinearity
    submarket_dummies = submarket_dummies.iloc[:, :-1]

    print(f"✅ Created {len(submarket_dummies.columns)} submarket indicator features")

    return submarket_dummies

def train_unified_submarket_model(data, base_features, test_cutoff='2022-12-31'):
    """
    Train single model with submarket features.

    Recommended when: Submarkets have <40 quarters of individual data
    """
    print("\n" + "="*80)
    print("APPROACH 2: UNIFIED MODEL WITH SUBMARKET FEATURES")
    print("="*80)

    # Create submarket features
    submarket_features = create_submarket_features(data)

    # Combine with base features
    X_full = pd.concat([
        data[base_features],
        submarket_features
    ], axis=1)

    all_features = list(X_full.columns)

    print(f"Total features: {len(all_features)}")
    print(f"  Base features: {len(base_features)}")
    print(f"  Submarket indicators: {len(submarket_features.columns)}")

    # Train/test split
    train_mask = data['date'] <= test_cutoff

    X_train = X_full[train_mask].fillna(method='ffill')
    y_train = data.loc[train_mask, 'rent_growth_yoy']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Split for validation
    n_train = int(len(X_scaled) * 0.8)
    X_tr, X_val = X_scaled[:n_train], X_scaled[n_train:]
    y_tr, y_val = y_train[:n_train], y_train[n_train:]

    # Train LightGBM
    print("\n[1/3] Training LightGBM...")
    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    lgb_model = lgb.train(
        PRODUCTION_CONFIG['lightgbm'],
        train_data,
        valid_sets=[valid_data],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=PRODUCTION_CONFIG['early_stopping']),
            lgb.log_evaluation(period=100)
        ]
    )

    # Note: SARIMA on unified model would need multi-series approach (VAR/VECM)
    # For simplicity, we'll use MSA-level SARIMA
    print("\n[2/3] Training MSA-level SARIMA (shared across submarkets)...")
    msa_rent_growth = data.groupby('date')['rent_growth_yoy'].mean()

    sarima_model = SARIMAX(
        msa_rent_growth,
        order=PRODUCTION_CONFIG['sarima']['order'],
        seasonal_order=PRODUCTION_CONFIG['sarima']['seasonal_order'],
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    # Train Ridge meta-learner
    print("\n[3/3] Training Ridge Meta-Learner...")
    lgb_pred = lgb_model.predict(X_scaled)

    # Align SARIMA predictions with training data
    sarima_pred = sarima_model.fittedvalues.values
    # Ensure same length
    sarima_pred = sarima_pred[-len(lgb_pred):]

    X_meta = np.column_stack([lgb_pred, sarima_pred])

    ridge_model = RidgeCV(
        alphas=PRODUCTION_CONFIG['ridge']['alphas'],
        cv=TimeSeriesSplit(n_splits=PRODUCTION_CONFIG['ridge']['cv'])
    ).fit(X_meta, y_train.values[-len(lgb_pred):])

    print(f"\n✅ Unified model trained")
    print(f"   Ridge alpha: {ridge_model.alpha_:.4f}")

    # Save models
    model_dir = 'models/unified_submarket'
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(lgb_model, f'{model_dir}/lightgbm_unified.pkl')
    joblib.dump(sarima_model, f'{model_dir}/sarima_msa.pkl')
    joblib.dump(ridge_model, f'{model_dir}/ridge_meta_unified.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler_unified.pkl')

    # Save feature list
    with open(f'{model_dir}/features.json', 'w') as f:
        json.dump(all_features, f, indent=2)

    print(f"✅ Models saved to {model_dir}/")

    return lgb_model, sarima_model, ridge_model, scaler, all_features

# ============================================================================
# FORECASTING FUNCTIONS
# ============================================================================

def generate_submarket_forecasts(models_dict, future_data, features,
                                 forecast_start='2026-01-01',
                                 forecast_end='2028-12-31'):
    """
    Generate forecasts for all submarkets using separate models.
    """
    print("\n" + "="*80)
    print(f"GENERATING SUBMARKET FORECASTS: {forecast_start} to {forecast_end}")
    print("="*80)

    all_forecasts = []

    for submarket, model in models_dict.items():
        print(f"\nForecasting: {submarket}")

        # Filter future data for this submarket
        sub_future = future_data[
            (future_data['submarket'] == submarket) &
            (future_data['date'] >= forecast_start) &
            (future_data['date'] <= forecast_end)
        ].copy()

        # Prepare features
        X_future = sub_future[features].fillna(method='ffill')

        # Generate predictions
        predictions = model.predict(X_future)

        # Create results dataframe
        results = pd.DataFrame({
            'date': sub_future['date'].values,
            'submarket': submarket,
            'ensemble_prediction': predictions
        })

        all_forecasts.append(results)

        print(f"  Forecast: {predictions.mean():.2f}% avg, "
              f"[{predictions.min():.2f}%, {predictions.max():.2f}%]")

    # Combine all forecasts
    combined = pd.concat(all_forecasts, ignore_index=True)

    print(f"\n{'='*80}")
    print(f"✅ Generated forecasts for {len(models_dict)} submarkets")
    print(f"{'='*80}")

    return combined

def rank_submarkets(forecast_df):
    """Rank submarkets by expected performance"""
    print("\n" + "="*80)
    print("SUBMARKET PERFORMANCE RANKING")
    print("="*80)

    # Calculate average forecast by submarket
    rankings = forecast_df.groupby('submarket').agg({
        'ensemble_prediction': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)

    rankings.columns = ['avg_growth', 'std_dev', 'min_growth', 'max_growth', 'quarters']
    rankings = rankings.sort_values('avg_growth', ascending=False)

    # Add ranking
    rankings['rank'] = range(1, len(rankings) + 1)

    print("\nSubmarket Rankings (2026-2028):")
    print(rankings.to_string())

    # Investment recommendations
    print("\n" + "-"*80)
    print("INVESTMENT RECOMMENDATIONS")
    print("-"*80)

    top_3 = rankings.head(3).index.tolist()
    bottom_3 = rankings.tail(3).index.tolist()

    print(f"\n✅ TOP PERFORMERS (Target for Acquisitions):")
    for i, submarket in enumerate(top_3, 1):
        avg = rankings.loc[submarket, 'avg_growth']
        print(f"   {i}. {submarket}: {avg:.2f}% avg growth")

    print(f"\n⚠️  UNDERPERFORMERS (Consider for Disposition):")
    for i, submarket in enumerate(bottom_3, 1):
        avg = rankings.loc[submarket, 'avg_growth']
        print(f"   {i}. {submarket}: {avg:.2f}% avg growth")

    return rankings

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_submarket_forecasts(forecast_df, rankings, output_dir='outputs/submarkets'):
    """Create comprehensive submarket forecast visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Phoenix Submarket Rent Growth Forecasts 2026-2028\n(Production-Validated Configuration)',
                 fontsize=16, fontweight='bold')

    # Plot 1: Top 5 submarkets over time
    ax1 = axes[0, 0]
    top_5 = rankings.head(5).index.tolist()

    for submarket in top_5:
        sub_data = forecast_df[forecast_df['submarket'] == submarket]
        ax1.plot(sub_data['date'], sub_data['ensemble_prediction'],
                marker='o', label=submarket, linewidth=2)

    ax1.set_title('Top 5 Performing Submarkets', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Rent Growth (%)')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Average growth ranking
    ax2 = axes[0, 1]
    rankings_sorted = rankings.sort_values('avg_growth')
    colors = ['green' if x > 3 else 'orange' if x > 2 else 'red'
              for x in rankings_sorted['avg_growth']]

    ax2.barh(range(len(rankings_sorted)), rankings_sorted['avg_growth'],
            color=colors, alpha=0.7)
    ax2.set_yticks(range(len(rankings_sorted)))
    ax2.set_yticklabels(rankings_sorted.index, fontsize=9)
    ax2.set_title('Average Rent Growth by Submarket (2026-2028)',
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('Average Rent Growth (%)')
    ax2.axvline(x=3, color='gray', linestyle='--', label='MSA Average (3.55%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Growth vs volatility
    ax3 = axes[1, 0]
    ax3.scatter(rankings['std_dev'], rankings['avg_growth'],
               s=100, alpha=0.6)

    for submarket in rankings.index:
        ax3.annotate(submarket,
                    (rankings.loc[submarket, 'std_dev'],
                     rankings.loc[submarket, 'avg_growth']),
                    fontsize=8, ha='right')

    ax3.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Volatility (Std Dev of Growth)')
    ax3.set_ylabel('Expected Return (Avg Growth %)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Quarterly distribution
    ax4 = axes[1, 1]
    forecast_df.boxplot(column='ensemble_prediction', by='submarket',
                       ax=ax4, rot=45)
    ax4.set_title('Forecast Distribution by Submarket',
                 fontsize=12, fontweight='bold')
    ax4.set_xlabel('Submarket')
    ax4.set_ylabel('Rent Growth (%)')
    plt.suptitle('')  # Remove automatic title

    plt.tight_layout()

    output_path = f"{output_dir}/submarket_forecasts_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Submarket forecast visualization saved: {output_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("="*80)
    print("PHOENIX SUBMARKET RENT GROWTH FORECAST FRAMEWORK")
    print("Production-Validated Methodology")
    print("="*80)
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Step 1: Generate/Load Data
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)

    # Generate template data (REPLACE with actual data loading)
    data = generate_template_submarket_data()
    save_template_data(data)

    # Define features (same as MSA model, excluding submarket column)
    base_features = [
        'fed_funds_rate', 'mortgage_rate_30yr', 'national_unemployment',
        'cpi', 'inflation_expectations_5yr', 'housing_starts',
        'building_permits', 'vacancy_rate', 'cap_rate',
        'phx_hpi_yoy_growth', 'phx_home_price_index', 'inventory_units',
        'units_under_construction', 'absorption_12mo',
        'phx_total_employment', 'phx_unemployment_rate',
        'phx_employment_yoy_growth', 'phx_prof_business_employment',
        'phx_manufacturing_employment', 'supply_inventory_ratio',
        'absorption_inventory_ratio', 'mortgage_employment_interaction',
        'migration_proxy', 'phx_prof_business_yoy_growth',
        'mortgage_rate_30yr_lag2'
    ]

    # Step 2: Choose Approach
    print("\n" + "="*80)
    print("STEP 2: MODEL SELECTION")
    print("="*80)
    print("\nTwo approaches available:")
    print("  1. Separate models per submarket (recommended if ≥40 quarters)")
    print("  2. Unified model with submarket features (better if <40 quarters)")

    # Check data availability per submarket
    data_counts = data.groupby('submarket').size()
    print(f"\nData availability per submarket:")
    print(data_counts)

    avg_quarters = data_counts.mean()
    print(f"\nAverage quarters per submarket: {avg_quarters:.0f}")

    if avg_quarters >= PRODUCTION_CONFIG['validation_thresholds']['min_data_quarters']:
        approach = "separate"
        print(f"\n✅ Recommended: SEPARATE MODELS (sufficient data)")
    else:
        approach = "unified"
        print(f"\n✅ Recommended: UNIFIED MODEL (limited data per submarket)")

    # Step 3: Train Models
    print("\n" + "="*80)
    print(f"STEP 3: TRAINING MODELS ({approach.upper()} APPROACH)")
    print("="*80)

    if approach == "separate":
        # Approach 1: Separate models
        submarket_models = train_separate_submarket_models(
            data, base_features, test_cutoff='2022-12-31'
        )

        # Generate forecasts
        future_data = data[data['date'] > '2025-12-31']

        forecasts = generate_submarket_forecasts(
            submarket_models, future_data, base_features,
            forecast_start='2026-01-01', forecast_end='2028-12-31'
        )

    else:
        # Approach 2: Unified model
        lgb_model, sarima_model, ridge_model, scaler, all_features = \
            train_unified_submarket_model(data, base_features, test_cutoff='2022-12-31')

        # Note: Unified approach forecasting would require additional implementation
        print("\n⚠️  Unified model trained successfully")
        print("   Forecasting from unified model requires additional implementation")
        print("   (predicting for each submarket with appropriate indicators)")

        forecasts = None

    # Step 4: Analysis & Rankings
    if forecasts is not None:
        print("\n" + "="*80)
        print("STEP 4: SUBMARKET ANALYSIS & RANKING")
        print("="*80)

        rankings = rank_submarkets(forecasts)

        # Save forecasts
        output_csv = f"outputs/submarkets/phoenix_submarket_forecasts_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs('outputs/submarkets', exist_ok=True)
        forecasts.to_csv(output_csv, index=False)
        print(f"\n✅ Submarket forecasts saved: {output_csv}")

        # Save rankings
        rankings_csv = f"outputs/submarkets/submarket_rankings_{datetime.now().strftime('%Y%m%d')}.csv"
        rankings.to_csv(rankings_csv)
        print(f"✅ Submarket rankings saved: {rankings_csv}")

        # Create visualizations
        plot_submarket_forecasts(forecasts, rankings)

    print("\n" + "="*80)
    print("FRAMEWORK DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Replace template data with actual submarket data")
    print("  2. Validate forecasts on historical test periods")
    print("  3. Use rankings for portfolio allocation decisions")
    print("  4. Update quarterly with new actuals")
    print("="*80)

if __name__ == '__main__':
    main()
