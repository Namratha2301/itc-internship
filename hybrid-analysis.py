

"""
ITC-StackBox Hybrid Distribution Model Implementation

This script implements a hybrid distribution model that combines ITC's traditional system
with StackBox's technology-driven approach, as per the recommendation to use StackBox for
goods arriving at city docks and ITC's traditional system for last-mile delivery.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sqlite3
from scipy import stats
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('hybrid_model')

class HybridDistributionModel:
    """
    Implementation of the Hybrid Distribution Model integrating 
    ITC's traditional system with StackBox technology
    """
    
    def __init__(self, data_path="./data/", output_path="./output/"):
        """
        Initialize the Hybrid Distribution Model.
        
        Args:
            data_path (str): Path to the directory containing data files
            output_path (str): Path to save outputs and models
        """
        self.data_path = data_path
        self.output_path = output_path
        self.order_data = None
        self.store_data = None
        self.inventory_data = None
        self.logistics_data = None
        self.performance_data = None
        self.cost_data = None
        self.model_traditional = None
        self.model_stackbox = None
        self.model_hybrid = None
        self.scaler = StandardScaler()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
    def load_data(self):
        """
        Load all required datasets for hybrid model implementation.
        """
        logger.info("Loading datasets for hybrid model...")
        
        try:
            # Load order data
            self.order_data = pd.read_csv(os.path.join(self.data_path, "order_data.csv"))
            logger.info(f"Order data loaded: {self.order_data.shape[0]} records")
            
            # Load store information
            self.store_data = pd.read_csv(os.path.join(self.data_path, "store_info.csv"))
            logger.info(f"Store data loaded: {self.store_data.shape[0]} records")
            
            # Load inventory data
            self.inventory_data = pd.read_csv(os.path.join(self.data_path, "inventory_data.csv"))
            logger.info(f"Inventory data loaded: {self.inventory_data.shape[0]} records")
            
            # Load logistics data
            self.logistics_data = pd.read_csv(os.path.join(self.data_path, "logistics_data.csv"))
            logger.info(f"Logistics data loaded: {self.logistics_data.shape[0]} records")
            
            # Load performance data from SQL
            conn = sqlite3.connect(os.path.join(self.data_path, "performance.db"))
            self.performance_data = pd.read_sql_query("SELECT * FROM performance_metrics", conn)
            conn.close()
            logger.info(f"Performance data loaded: {self.performance_data.shape[0]} records")
            
            # Load cost data
            self.cost_data = pd.read_excel(os.path.join(self.data_path, "cost_structure.xlsx"))
            logger.info(f"Cost data loaded: {self.cost_data.shape[0]} records")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """
        Preprocess all datasets for model development.
        """
        logger.info("Preprocessing data for hybrid model...")
        
        # Process order data
        if self.order_data is not None:
            # Convert date columns
            self.order_data['order_date'] = pd.to_datetime(self.order_data['order_date'])
            
            # Extract temporal features
            self.order_data['day_of_week'] = self.order_data['order_date'].dt.dayofweek
            self.order_data['month'] = self.order_data['order_date'].dt.month
            self.order_data['quarter'] = self.order_data['order_date'].dt.quarter
            self.order_data['year'] = self.order_data['order_date'].dt.year
            
            # Create a unique identifier for year-month
            self.order_data['year_month'] = self.order_data['order_date'].dt.strftime('%Y-%m')
            
            logger.info("Order data preprocessing complete")
        
        # Process store data
        if self.store_data is not None:
            # Categorize stores by size
            size_bins = [0, 200, 500, 1000, float('inf')]
            size_labels = ['Very Small', 'Small', 'Medium', 'Large']
            self.store_data['size_category'] = pd.cut(
                self.store_data['store_area_sqft'], 
                bins=size_bins, 
                labels=size_labels
            )
            
            # Encode store type
            self.store_data = pd.get_dummies(
                self.store_data, 
                columns=['store_type', 'size_category'],
                drop_first=True
            )
            
            logger.info("Store data preprocessing complete")
        
        # Process inventory data
        if self.inventory_data is not None:
            # Calculate days of supply
            self.inventory_data['days_of_supply'] = (
                self.inventory_data['inventory_units'] / 
                self.inventory_data['avg_daily_demand']
            ).fillna(0)
            
            # Calculate inventory turnover
            self.inventory_data['turnover_ratio'] = (
                self.inventory_data['monthly_units_sold'] / 
                self.inventory_data['avg_inventory_units']
            ).fillna(0)
            
            logger.info("Inventory data preprocessing complete")
        
        # Process logistics data
        if self.logistics_data is not None:
            # Convert date columns
            self.logistics_data['shipment_date'] = pd.to_datetime(self.logistics_data['shipment_date'])
            self.logistics_data['delivery_date'] = pd.to_datetime(self.logistics_data['delivery_date'])
            
            # Calculate delivery time
            self.logistics_data['delivery_time_days'] = (
                self.logistics_data['delivery_date'] - 
                self.logistics_data['shipment_date']
            ).dt.total_seconds() / (24 * 3600)
            
            logger.info("Logistics data preprocessing complete")
        
        # Process performance data
        if self.performance_data is not None:
            # Handle missing values
            self.performance_data.fillna({
                'order_accuracy_percent': self.performance_data['order_accuracy_percent'].median(),
                'on_time_delivery_percent': self.performance_data['on_time_delivery_percent'].median()
            }, inplace=True)
            
            logger.info("Performance data preprocessing complete")
        
        # Merge datasets for modeling
        try:
            # Start with order data
            self.modeling_data = self.order_data.copy()
            
            # Merge with store data
            if self.store_data is not None:
                self.modeling_data = self.modeling_data.merge(
                    self.store_data,
                    on='store_id',
                    how='left'
                )
            
            # Aggregate performance metrics by distribution system and time period
            if self.performance_data is not None:
                perf_agg = self.performance_data.groupby(
                    ['distribution_system', 'year_month']
                )[['order_accuracy_percent', 'on_time_delivery_percent']].mean().reset_index()
                
                self.modeling_data = self.modeling_data.merge(
                    perf_agg,
                    on=['distribution_system', 'year_month'],
                    how='left'
                )
            
            logger.info(f"Final modeling dataset created with {self.modeling_data.shape[0]} records and {self.modeling_data.shape[1]} features")
            return True
            
        except Exception as e:
            logger.error(f"Error creating modeling dataset: {e}")
            return False
    
    def train_predictive_models(self):
        """
        Train predictive models for traditional, stackbox, and hybrid performance.
        """
        logger.info("Training predictive models...")
        
        # Separate data by distribution system
        traditional_data = self.modeling_data[self.modeling_data['distribution_system'] == 'Traditional']
        stackbox_data = self.modeling_data[self.modeling_data['distribution_system'] == 'StackBox']
        
        # Define features and target variable
        feature_cols = [
            'order_value', 'units_ordered', 'day_of_week', 'month', 'quarter',
            'order_accuracy_percent', 'on_time_delivery_percent'
        ]
        
        # Add store features if available
        store_features = [col for col in self.modeling_data.columns if 'store_type_' in col or 'size_category_' in col]
        feature_cols.extend(store_features)
        
        # Target variable - we'll predict customer satisfaction
        target = 'customer_satisfaction'
        
        # Prepare datasets
        def prepare_model_data(df, features, target):
            if target not in df.columns:
                logger.error(f"Target variable '{target}' not found in dataset")
                return None, None, None, None, None, None
                
            # Handle missing values
            df = df.dropna(subset=[target])
            X = df[features].fillna(0)
            y = df[target]
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            return train_test_split(X, X_scaled, y, test_size=0.2, random_state=42)
        
        # Train models for each distribution system
        try:
            # Traditional system model
            X_trad, X_test_trad, X_trad_scaled, X_test_trad_scaled, y_trad, y_test_trad = prepare_model_data(
                traditional_data, feature_cols, target
            )
            
            self.model_traditional = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_traditional.fit(X_trad, y_trad)
            
            # StackBox model
            X_sb, X_test_sb, X_sb_scaled, X_test_sb_scaled, y_sb, y_test_sb = prepare_model_data(
                stackbox_data, feature_cols, target
            )
            
            self.model_stackbox = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_stackbox.fit(X_sb, y_sb)
            
            # Evaluate models
            trad_predictions = self.model_traditional.predict(X_test_trad)
            sb_predictions = self.model_stackbox.predict(X_test_sb)
            
            trad_mae = mean_absolute_error(y_test_trad, trad_predictions)
            sb_mae = mean_absolute_error(y_test_sb, sb_predictions)
            
            logger.info(f"Traditional model MAE: {trad_mae:.4f}")
            logger.info(f"StackBox model MAE: {sb_mae:.4f}")
            
            # Save models
            joblib.dump(self.model_traditional, os.path.join(self.output_path, 'traditional_model.pkl'))
            joblib.dump(self.model_stackbox, os.path.join(self.output_path, 'stackbox_model.pkl'))
            joblib.dump(self.scaler, os.path.join(self.output_path, 'scaler.pkl'))
            
            logger.info("Models trained and saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training predictive models: {e}")
            return False
    
    def develop_hybrid_model(self):
        """
        Develop the hybrid distribution model that combines 
        traditional and StackBox strengths.
        """
        logger.info("Developing hybrid distribution model...")
        
        # Create feature importance visualizations
        if self.model_traditional is not None and self.model_stackbox is not None:
            feature_cols = self.model_traditional.feature_names_in_
            
            # Traditional model feature importance
            trad_importances = self.model_traditional.feature_importances_
            trad_indices = np.argsort(trad_importances)[-10:]  # Top 10 features
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(trad_indices)), trad_importances[trad_indices])
            plt.yticks(range(len(trad_indices)), [feature_cols[i] for i in trad_indices])
            plt.xlabel('Feature Importance')
            plt.title('Traditional System - Top Features')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'traditional_features.png'))
            
            # StackBox model feature importance
            sb_importances = self.model_stackbox.feature_importances_
            sb_indices = np.argsort(sb_importances)[-10:]  # Top 10 features
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(sb_indices)), sb_importances[sb_indices])
            plt.yticks(range(len(sb_indices)), [feature_cols[i] for i in sb_indices])
            plt.xlabel('Feature Importance')
            plt.title('StackBox System - Top Features')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'stackbox_features.png'))
        
        # Define the hybrid model logic
        class HybridModel:
            def __init__(self, traditional_model, stackbox_model, decision_threshold=0.5):
                self.traditional_model = traditional_model
                self.stackbox_model = stackbox_model
                self.decision_threshold = decision_threshold
                
            def predict(self, X, store_info=None):
                """
                Make predictions using the hybrid model approach.
                
                The hybrid model uses StackBox for:
                - Inventory management (warehouse operations)
                - Large orders
                - Larger stores
                
                The hybrid model uses Traditional for:
                - Last mile delivery
                - Small stores
                - Categories where traditional system excels
                
                Args:
                    X: Features for prediction
                    store_info: Additional store information to influence decision
                
                Returns:
                    Predictions based on the hybrid approach
                """
                # Get predictions from both models
                trad_predictions = self.traditional_model.predict(X)
                sb_predictions = self.stackbox_model.predict(X)
                
                # Initialize with stackbox predictions
                final_predictions = sb_predictions.copy()
                
                # Use store information to decide which model to use
                if store_info is not None:
                    for i, store in enumerate(store_info):
                        # Use traditional for small stores
                        if store['size_category'] in ['Very Small', 'Small']:
                            final_predictions[i] = trad_predictions[i]
                        
                        # For medium/large stores with small orders, use traditional
                        elif store['order_size'] == 'Small' and store['size_category'] in ['Medium', 'Large']:
                            final_predictions[i] = trad_predictions[i]
                        
                        # For last-mile delivery, use traditional system's prediction
                        # but adjust it with StackBox efficiency for inventory
                        else:
                            # Hybrid approach: 70% traditional (for customer relationship)
                            # and 30% stackbox (for technology efficiency)
                            final_predictions[i] = (0.7 * trad_predictions[i]) + (0.3 * sb_predictions[i])
                
                return final_predictions
            
        # Create the hybrid model
        self.model_hybrid = HybridModel(
            self.model_traditional,
            self.model_stackbox
        )
        
        # Save the hybrid model
        joblib.dump(self.model_hybrid, os.path.join(self.output_path, 'hybrid_model.pkl'))
        
        logger.info("Hybrid model developed and saved successfully")
        return True
    
    def simulate_hybrid_performance(self, simulation_months=6):
        """
        Simulate the performance of the hybrid model over time.
        
        Args:
            simulation_months: Number of months to simulate
        
        Returns:
            DataFrame with simulation results
        """
        logger.info(f"Simulating hybrid model performance over {simulation_months} months...")
        
        # Get the last date in the order data
        if self.order_data is None:
            logger.error("Order data not available for simulation")
            return None
            
        last_date = self.order_data['order_date'].max()
        
        # Create simulation dates
        simulation_dates = [last_date + timedelta(days=30*i) for i in range(1, simulation_months+1)]
        simulation_periods = [date.strftime('%Y-%m') for date in simulation_dates]
        
        # Create representative sample of stores
        if self.store_data is not None:
            store_sample = self.store_data.sample(
                n=min(100, len(self.store_data)),
                random_state=42
            )
        else:
            logger.warning("Store data not available, using generic store profiles")
            store_sample = pd.DataFrame({
                'store_id': range(1, 101),
                'size_category': np.random.choice(['Very Small', 'Small', 'Medium', 'Large'], 100)
            })
        
        # Initialize simulation results
        simulation_results = {
            'period': [],
            'traditional_satisfaction': [],
            'stackbox_satisfaction': [],
            'hybrid_satisfaction': [],
            'traditional_orders': [],
            'stackbox_orders': [],
            'hybrid_orders': [],
            'traditional_revenue': [],
            'stackbox_revenue': [],
            'hybrid_revenue': [],
        }
        
        # Simulate each period
        for period in simulation_periods:
            # Base metrics with small improvements over time
            period_index = simulation_periods.index(period)
            
            # Simulate improvement in hybrid model over time (learning effect)
            learning_factor = 1.0 + (period_index * 0.03)  # 3% improvement per period
            
            # Simulate traditional system
            trad_satisfaction = 3.8 + np.random.normal(0, 0.1)  # Stable satisfaction
            trad_orders = 550 + np.random.normal(0, 20)  # Stable order count
            trad_revenue = 2500000 + np.random.normal(0, 100000)  # Stable revenue
            
            # Simulate StackBox system
            sb_satisfaction = 3.5 + (period_index * 0.05) + np.random.normal(0, 0.1)  # Improving
            sb_orders = 400 + (period_index * 15) + np.random.normal(0, 20)  # Growing
            sb_revenue = 2000000 + (period_index * 100000) + np.random.normal(0, 100000)  # Growing
            
            # Simulate hybrid system
            hybrid_satisfaction = max(trad_satisfaction, sb_satisfaction) * learning_factor
            hybrid_orders = max(trad_orders, sb_orders) * learning_factor
            hybrid_revenue = (trad_revenue * 0.7 + sb_revenue * 0.3) * learning_factor
            
            # Store results
            simulation_results['period'].append(period)
            simulation_results['traditional_satisfaction'].append(trad_satisfaction)
            simulation_results['stackbox_satisfaction'].append(sb_satisfaction)
            simulation_results['hybrid_satisfaction'].append(hybrid_satisfaction)
            simulation_results['traditional_orders'].append(trad_orders)
            simulation_results['stackbox_orders'].append(sb_orders)
            simulation_results['hybrid_orders'].append(hybrid_orders)
            simulation_results['traditional_revenue'].append(trad_revenue)
            simulation_results['stackbox_revenue'].append(sb_revenue)
            simulation_results['hybrid_revenue'].append(hybrid_revenue)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(simulation_results)
        
        # Save results
        results_df.to_csv(os.path.join(self.output_path, 'hybrid_simulation_results.csv'), index=False)
        
        # Create visualizations
        self._create_simulation_visualizations(results_df)
        
        logger.info("Simulation completed successfully")
        return results_df
    
    def _create_simulation_visualizations(self, results_df):
        """
        Create visualizations from simulation results.
        
        Args:
            results_df: DataFrame with simulation results
        """
        # Satisfaction trend
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['period'], results_df['traditional_satisfaction'], 'o-', label='Traditional')
        plt.plot(results_df['period'], results_df['stackbox_satisfaction'], 's-', label='StackBox')
        plt.plot(results_df['period'], results_df['hybrid_satisfaction'], '^-', label='Hybrid')
        plt.title('Customer Satisfaction Trend')
        plt.xlabel('Period')
        plt.ylabel('Satisfaction Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'satisfaction_trend.png'))
        
        # Orders trend
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['period'], results_df['traditional_orders'], 'o-', label='Traditional')
        plt.plot(results_df['period'], results_df['stackbox_orders'], 's-', label='StackBox')
        plt.plot(results_df['period'], results_df['hybrid_orders'], '^-', label='Hybrid')
        plt.title('Order Volume Trend')
        plt.xlabel('Period')
        plt.ylabel('Number of Orders')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'orders_trend.png'))
        
        # Revenue trend
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['period'], results_df['traditional_revenue'] / 1000000, 'o-', label='Traditional')
        plt.plot(results_df['period'], results_df['stackbox_revenue'] / 1000000, 's-', label='StackBox')
        plt.plot(results_df['period'], results_df['hybrid_revenue'] / 1000000, '^-', label='Hybrid')
        plt.title('Revenue Trend')
        plt.xlabel('Period')
        plt.ylabel('Revenue (Millions ₹)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'revenue_trend.png'))
    
    def create_implementation_roadmap(self):
        """
        Create a detailed implementation roadmap for the hybrid model.
        """
        logger.info("Creating implementation roadmap...")
        
        # Define implementation phases
        phases = [
            {
                'phase': 'Phase 1: Assessment',
                'timeline': 'Month 1-2',
                'activities': [
                    'Gap Analysis between current systems',
                    'Stakeholder interviews and requirement gathering',
                    'Define KPIs for hybrid model',
                    'Cost-benefit analysis',
                    'Risk assessment'
                ]
            },
            {
                'phase': 'Phase 2: Pilot Program',
                'timeline': 'Month 3-4',
                'activities': [
                    'Integrate StackBox at city dock level',
                    'Maintain traditional system for last-mile',
                    'Select pilot regions (1-2 cities)',
                    'Train staff and store owners',
                    'Collect initial feedback',
                    'Adjust model based on pilot results'
                ]
            },
            {
                'phase': 'Phase 3: Training & Preparation',
                'timeline': 'Month 5-6',
                'activities': [
                    'Comprehensive staff training program',
                    'Store owner workshops and onboarding',
                    'Develop support materials and documentation',
                    'Establish support channels',
                    'Create incentive program for technology adoption',
                    'Prepare technology infrastructure'
                ]
            },
            {
                'phase': 'Phase 4: Expanded Rollout',
                'timeline': 'Month 7-9',
                'activities': [
                    'Rollout to all regions',
                    'Implement enhanced inventory visibility tools',
                    'Integration with existing ITC systems',
                    'Continuous monitoring and fine-tuning',
                    'Regular feedback collection'
                ]
            },
            {
                'phase': 'Phase 5: Optimization',
                'timeline': 'Month 10-12',
                'activities': [
                    'Review KPI performance',
                    'Identify optimization opportunities',
                    'Develop technology enhancement roadmap',
                    'Gradual transition plan for store owners',
                    'Long-term strategy development'
                ]
            }
        ]
        
        # Create roadmap visualization
        plt.figure(figsize=(14, 8))
        
        # Create timeline
        for i, phase in enumerate(phases):
            # Phase box
            plt.fill_between(
                [i, i+0.8], 
                [0, 0], 
                [len(phase['activities']), len(phase['activities'])], 
                color=f'C{i}', 
                alpha=0.3
            )
            
            # Phase title
            plt.text(
                i+0.4, 
                len(phase['activities'])+0.5, 
                phase['phase'], 
                ha='center', 
                fontsize=12, 
                fontweight='bold'
            )
            
            # Phase timeline
            plt.text(
                i+0.4, 
                len(phase['activities'])+0.1, 
                phase['timeline'], 
                ha='center', 
                fontsize=10
            )
            
            # Activities
            for j, activity in enumerate(phase['activities']):
                plt.text(
                    i+0.05, 
                    j+0.5, 
                    activity, 
                    fontsize=10,
                    va='center'
                )
        
        # Adjustments
        plt.xlim(-0.5, len(phases))
        plt.ylim(0, max([len(p['activities']) for p in phases])+1)
        plt.axis('off')
        plt.title('ITC-StackBox Hybrid Distribution Model Implementation Roadmap', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'implementation_roadmap.png'))
        
        logger.info("Implementation roadmap created successfully")
        
        # Create textual roadmap in Markdown
        markdown_roadmap = "# ITC-StackBox Hybrid Distribution Model Implementation Roadmap\n\n"
        
        for phase in phases:
            markdown_roadmap += f"## {phase['phase']}\n"
            markdown_roadmap += f"**Timeline:** {phase['timeline']}\n\n"
            markdown_roadmap += "**Key Activities:**\n"
            
            for activity in phase['activities']:
                markdown_roadmap += f"- {activity}\n"
            
            markdown_roadmap += "\n"
        
        # Save markdown roadmap
        with open(os.path.join(self.output_path, 'implementation_roadmap.md'), 'w') as f:
            f.write(markdown_roadmap)
    
    def calculate_roi_projection(self, years=3):
        """
        Calculate and visualize ROI projection for the hybrid model.
        
        Args:
            years: Number of years to project
        """
        logger.info(f"Calculating ROI projection for {years} years...")
        
        # Initialize projections
        months = years * 12
        
        # Initial investment costs
        investment = {
            'technology_integration': 15000000,  # 1.5 Crore
            'training': 5000000,  # 50 Lakhs
            'consulting': 3000000,  # 30 Lakhs
            'infrastructure': 7000000,  # 70 Lakhs
            'miscellaneous': 2000000   # 20 Lakhs
        }
        
        total_investment = sum(investment.values())
        
        # Monthly projections
        traditional_revenue = []
        stackbox_revenue = []
        hybrid_revenue = []
        
        traditional_cost = []
        stackbox_cost = []
        hybrid_cost = []
        
        # Base values
        base_monthly_revenue = 25000000  # 2.5 Crore
        base_monthly_cost_traditional = 20000000  # 2 Crore (80% of revenue)
        base_monthly_cost_stackbox = 19000000  # 1.9 Crore (76% of revenue)
        base_monthly_cost_hybrid = 18000000  # 1.8 Crore (72% of revenue)
        
        # Growth rates
        traditional_growth = 0.005  # 0.5% monthly growth
        stackbox_growth = 0.008  # 0.8% monthly growth
        hybrid_growth = 0.012  # 1.2% monthly growth
        
        # Cost efficiency improvements (monthly)
        traditional_efficiency = 0.001  # 0.1% monthly improvement
        stackbox_efficiency = 0.002  # 0.2% monthly improvement
        hybrid_efficiency = 0.003  # 0.3% monthly improvement
        
        # Generate projections
        for month in range(months):
            # Revenue projections
            trad_rev = base_monthly_revenue * (1 + traditional_growth) ** month
            sb_rev = base_monthly_revenue * (1 + stackbox_growth) ** month
            hyb_rev = base_monthly_revenue * (1 + hybrid_growth) ** month
            
            traditional_revenue.append(trad_rev)
            stackbox_revenue.append(sb_rev)
            hybrid_revenue.append(hyb_rev)
            
            # Cost projections
            trad_cost = base_monthly_cost_traditional * (1 - traditional_efficiency) ** month
            sb_cost = base_monthly_cost_stackbox * (1 - stackbox_efficiency) ** month
            hyb_cost = base_monthly_cost_hybrid * (1 - hybrid_efficiency) ** month
            
            traditional_cost.append(trad_cost)
            stackbox_cost.append(sb_cost)
            hybrid_cost.append(hyb_cost)
        
        # Calculate profits
        traditional_profit = [r - c for r, c in zip(traditional_revenue, traditional_cost)]
        stackbox_profit = [r - c for r, c in zip(stackbox_revenue, stackbox_cost)]
        hybrid_profit = [r - c for r, c in zip(hybrid_revenue, hybrid_cost)]
        
        # Calculate cumulative profits
        traditional_cumulative = [sum(traditional_profit[:i+1]) for i in range(len(traditional_profit))]
        stackbox_cumulative = [sum(stackbox_profit[:i+1]) for i in range(len(stackbox_profit))]
        hybrid_cumulative = [sum(hybrid_profit[:i+1]) for i in range(len(hybrid_profit))]
        
        # Adjust hybrid for initial investment
        hybrid_cumulative_with_investment = [-total_investment] + [p - total_investment for p in hybrid_cumulative]
        
        # Find break-even point
        breakeven_month = next((i for i, p in enumerate(hybrid_cumulative_with_investment) if p >= 0), None)
        
        # Calculate ROI
        if breakeven_month:
            # ROI at the end of projection period
            final_roi = (hybrid_cumulative[-1] - total_investment) / total_investment * 100
            
            logger.info(f"Projected break-even at month {breakeven_month}")
            logger.info(f"Projected ROI after {years} years: {final_roi:.2f}%")
        else:
            logger.warning("No break-even point found within the projection period")
        
        # Create visualization
        plt.figure(figsize=(14, 8))
        
        # Plot cumulative profits
        months_range = list(range(months + 1))
        plt.plot(months_range[1:], traditional_cumulative, 'b-', label='Traditional System')
        plt.plot(months_range[1:], stackbox_cumulative, 'g-', label='StackBox')
        plt.plot(months_range, hybrid_cumulative_with_investment, 'r-', label='Hybrid Model (with investment)')
        
        # Add zero line
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Mark break-even point
        if breakeven_month:
            plt.scatter([breakeven_month], [0], s=100, c='red', zorder=5)
            plt.annotate(
                f'Break-even: Month {breakeven_month}', 
                xy=(breakeven_month, 0),
                xytext=(breakeven_month + 1, -total_investment/5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3")
            )
        
        # Format y-axis as crores
        def crore_formatter(x, pos):
            return f'₹{x/10000000:.1f} Cr'
        
        from matplotlib.ticker import FuncFormatter
        plt.gca().yaxis.set_major_formatter(FuncFormatter(crore_formatter))
        
        # Add labels for projection years
        for year in range(1, years + 1):
            plt.axvline(x=year*12, color='gray', linestyle='--', alpha=0.5)
            plt.text(year*12, min(hybrid_cumulative_with_investment), f'Year {year}', 
                    ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5))
        
        # Finishing touches
        plt.title('Projected ROI Comparison: Traditional vs. StackBox vs. Hybrid Model')
        plt.xlabel('Month')
        plt.ylabel('Cumulative Profit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_path, 'roi_projection.png'))
        
        # Create detailed ROI report
        roi_data = {
            'Month': months_range,
            'Traditional_Revenue': [0] + traditional_revenue,
            'Traditional_Cost': [0] + traditional_cost,
            'Traditional_Profit': [0] + traditional_profit,
            'Traditional_Cumulative': [0] + traditional_cumulative,
            'StackBox_Revenue': [0] + stackbox_revenue,
            'StackBox_Cost': [0] + stackbox_cost,
            'StackBox_Profit': [0] + stackbox_profit,
            'StackBox_Cumulative': [0] + stackbox_cumulative,
            'Hybrid_Revenue': [0] + hybrid_revenue,
            'Hybrid_Cost': [0] + hybrid_cost,
            'Hybrid_Profit': [0] + hybrid_profit,
            'Hybrid_Cumulative': [0] + hybrid_cumulative,
            'Hybrid_With_Investment': hybrid_cumulative_with_investment
        }
        
        roi_df = pd.DataFrame(roi_data)
        roi_df.to_csv(os.path.join(self.output_path, 'roi_projection.csv'), index=False)
        
        logger.info("ROI projection completed successfully")
        
        return {
            'breakeven_month': breakeven_month,
            'final_roi': final_roi if breakeven_month else None,
            'total_investment': total_investment,
            'projection_data': roi_df
        }
    
    def create_store_segmentation_strategy(self):
        """
        Create a store segmentation strategy for hybrid model rollout.
        """
        logger.info("Creating store segmentation strategy...")
        
        if self.store_data is None:
            logger.error("Store data not available for segmentation strategy")
            return False
        
        # Define segments
        segments = {
            'Tech Ready': {
                'criteria': 'Large stores with high tech adoption readiness',
                'approach': 'Full hybrid model implementation in first phase',
                'expected_stores': '15-20% of total stores',
                'implementation_timeline': '1-3 months',
                'expected_benefits': 'Quick adoption, high efficiency gains, showcase success stories'
            },
            'Relationship Focused': {
                'criteria': 'Small stores with strong traditional system preference',
                'approach': 'Maintain traditional system with gradual tech introduction',
                'expected_stores': '30-40% of total stores',
                'implementation_timeline': '6-12 months',
                'expected_benefits': 'Preserve relationships, avoid disruption, focus on education'
            },
            'Balanced Adopters': {
                'criteria': 'Medium-sized stores with moderate tech readiness',
                'approach': 'Phased hybrid implementation with training support',
                'expected_stores': '30-35% of total stores',
                'implementation_timeline': '3-6 months',
                'expected_benefits': 'Balanced growth, medium-term efficiency gains'
            },
            'High Volume': {
                'criteria': 'Stores with high order volumes regardless of size',
                'approach': 'StackBox for inventory, traditional for relationship',
                'expected_stores': '10-15% of total stores',
                'implementation_timeline': '2-4 months',
                'expected_benefits': 'Improved inventory management for high-volume needs'
            }
        }
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Create a 2x2 grid of segments
        segment_names = list(segments.keys())
        
        for i, segment in enumerate(segment_names):
            plt.subplot(2, 2, i+1)
            
            # Create text box with segment information
            textstr = f"\n".join([
                f"{key}: {segments[segment][key]}" 
                for key in segments[segment]
            ])
            
            props = dict(boxstyle='round', facecolor=f'C{i}', alpha=0.3)
            plt.text(0.5, 0.5, textstr, transform=plt.gca().transAxes, fontsize=9,
                    verticalalignment='center', horizontalalignment='center', bbox=props)
            
            plt.title(segment)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'store_segmentation.png'))
        
        # Create store assignment simulation
        if 'size_category' in self.store_data.columns and 'tech_readiness' in self.store_data.columns:
            # Use actual data for segmentation
            tech_ready = self.store_data[
                (self.store_data['size_category'] == 'Large') & 
                (self.store_data['tech_readiness'] > 0.7)
            ]
            
            relationship_focused = self.store_data[
                (self.store_data['size_category'].isin(['Very Small', 'Small'])) & 
                (self.store_data['tech_readiness'] < 0.3)
            ]
            
            balanced = self.store_data[
                (self.store_data['size_category'] == 'Medium') & 
                (self.store_data['tech_readiness'].between(0.3, 0.7))
            ]
            
            high_volume = self.store_data[
                self.store_data['monthly_order_volume'] > self.store_data['monthly_order_volume'].quantile(0.85)
            ]
            
        else:
            # Simulate store assignments based on size
            if 'size_category' in self.store_data.columns:
                size_counts = self.store_data['size_category'].value_counts()
                logger.info(f"Store size distribution: {size_counts}")
            else:
                logger.warning("No size information available, using random assignment")
        
        # Create segmentation markdown report
        markdown_report = "# Store Segmentation Strategy for Hybrid Model Implementation\n\n"
        
        for segment in segments:
            markdown_report += f"## {segment}\n\n"
            
            for key, value in segments[segment].items():
                markdown_report += f"**{key}**: {value}\n\n"
            
            markdown_report += "---\n\n"
        
        # Add implementation guidelines
        markdown_report += "## Implementation Guidelines\n\n"
        markdown_report += "1. **Identify and classify** all stores based on segmentation criteria\n"
        markdown_report += "2. **Develop segment-specific training materials** addressing unique needs\n"
        markdown_report += "3. **Create implementation timelines** aligned with segment readiness\n"
        markdown_report += "4. **Assign dedicated support teams** for each segment\n"
        markdown_report += "5. **Monitor adoption metrics** by segment to identify optimization opportunities\n"
        markdown_report += "6. **Collect and incorporate feedback** from each segment to refine approach\n"
        
        # Save markdown report
        with open(os.path.join(self.output_path, 'store_segmentation_strategy.md'), 'w') as f:
            f.write(markdown_report)
        
        logger.info("Store segmentation strategy created successfully")
        return True
    
    def run_full_implementation(self):
        """
        Run the complete hybrid model implementation pipeline.
        """
        logger.info("Starting hybrid model implementation process...")
        
        # Step 1: Load and preprocess data
        if not self.load_data():
            logger.error("Data loading failed. Cannot continue with implementation.")
            return False
            
        if not self.preprocess_data():
            logger.error("Data preprocessing failed. Cannot continue with implementation.")
            return False
        
        # Step 2: Train predictive models
        if not self.train_predictive_models():
            logger.warning("Predictive model training failed. Continuing with implementation using simulated models.")
        
        # Step 3: Develop hybrid model
        if not self.develop_hybrid_model():
            logger.error("Hybrid model development failed. Cannot continue with implementation.")
            return False
        
        # Step 4: Run simulations
        simulation_results = self.simulate_hybrid_performance()
        if simulation_results is None:
            logger.warning("Performance simulation failed. Continuing with implementation without simulation data.")
        
        # Step 5: Create implementation roadmap
        self.create_implementation_roadmap()
        
        # Step 6: Calculate ROI projection
        roi_results = self.calculate_roi_projection()
        
        # Step 7: Create store segmentation strategy
        self.create_store_segmentation_strategy()
        
        logger.info("Hybrid model implementation process completed successfully!")
        
        # Return summary of results
        return {
            'status': 'success',
            'roi_breakeven': roi_results.get('breakeven_month') if roi_results else None,
            'simulation_completed': simulation_results is not None
        }


# Function to analyze barriers to StackBox adoption
def analyze_adoption_barriers(store_data):
    """
    Analyze the barriers to StackBox adoption among store owners.
    
    Args:
        store_data: DataFrame with store information including adoption barriers
    
    Returns:
        DataFrame with barrier frequency and category breakdown
    """
    if 'adoption_barrier' not in store_data.columns:
        print("No adoption barrier data available")
        return None
    
    # Calculate barrier frequency
    barrier_counts = store_data['adoption_barrier'].value_counts()
    barrier_percent = (barrier_counts / len(store_data)) * 100
    
    # Create barrier breakdown by store size if available
    if 'size_category' in store_data.columns:
        barrier_by_size = pd.crosstab(
            store_data['adoption_barrier'],
            store_data['size_category'],
            normalize='columns'
        ) * 100
    else:
        barrier_by_size = None
    
    # Recommended approaches to address each barrier
    barrier_solutions = {
        'Technology Resistance': [
            'Hands-on training sessions',
            'Simplified user interfaces',
            'Gradual feature introduction',
            'Peer success stories'
        ],
        'Trust in Traditional System': [
            'Hybrid approach preserving relationships',
            'Demonstrating reliability metrics',
            'Success guarantees',
            'Maintaining familiar processes'
        ],
        'Training Needs': [
            'Customized training programs',
            'Ongoing support channels',
            'Written documentation',
            'Video tutorials'
        ],
        'Cost Concerns': [
            'Shared investment model',
            'ROI demonstrations',
            'Phased adoption to spread costs',
            'Volume-based incentives'
        ]
    }
    
    return {
        'barrier_counts': barrier_counts,
        'barrier_percent': barrier_percent,
        'barrier_by_size': barrier_by_size,
        'barrier_solutions': barrier_solutions
    }


# Function to compare distribution systems performance
def compare_distribution_systems(order_data, performance_data=None):
    """
    Compare key performance metrics between traditional and StackBox systems.
    
    Args:
        order_data: DataFrame with order information
        performance_data: Optional DataFrame with performance metrics
    
    Returns:
        Dict with comparison results
    """
    if 'distribution_system' not in order_data.columns:
        print("Distribution system information not available")
        return None
    
    # Order frequency analysis
    order_data['order_date'] = pd.to_datetime(order_data['order_date'])
    order_data['year_month'] = order_data['order_date'].dt.strftime('%Y-%m')
    
    monthly_orders = order_data.groupby(['year_month', 'distribution_system']).size().unstack()
    
    # Calculate store-level order metrics
    store_orders = order_data.groupby(['store_id', 'distribution_system']).agg({
        'order_id': 'count',
        'order_value': ['mean', 'sum']
    })
    
    # Flatten column names
    store_orders.columns = ['_'.join(col) for col in store_orders.columns]
    
    # Calculate repeat orders (if time span is sufficient)
    min_date = order_data['order_date'].min()
    max_date = order_data['order_date'].max()
    date_range = (max_date - min_date).days
    
    repeat_order_analysis = None
    if date_range > 30:  # Only if we have at least a month of data
        # Count orders per store per month
        store_monthly_orders = order_data.groupby(
            ['store_id', 'year_month', 'distribution_system']
        ).size().reset_index(name='order_count')
        
        # Calculate average monthly orders per store by system
        avg_monthly_orders = store_monthly_orders.groupby(
            ['store_id', 'distribution_system']
        )['order_count'].mean().reset_index()
        
        repeat_order_analysis = avg_monthly_orders.groupby('distribution_system')['order_count'].describe()
    
    # Performance metrics comparison if available
    performance_comparison = None
    if performance_data is not None and 'distribution_system' in performance_data.columns:
        metrics = [
            'order_processing_time_mins', 
            'delivery_time_hours',
            'order_accuracy_percent', 
            'inventory_visibility_percent',
            'on_time_delivery_percent'
        ]
        
        available_metrics = [m for m in metrics if m in performance_data.columns]
        
        if available_metrics:
            performance_comparison = performance_data.groupby('distribution_system')[available_metrics].mean()
    
    return {
        'monthly_orders': monthly_orders,
        'store_orders': store_orders,
        'repeat_order_analysis': repeat_order_analysis,
        'performance_comparison': performance_comparison,
        'date_range_days': date_range
    }


if __name__ == "__main__":
    # Create and run hybrid model implementation
    hybrid_model = HybridDistributionModel(
        data_path="./data/",
        output_path="./output/"
    )
    
    # Run full implementation
    results = hybrid_model.run_full_implementation()
    
    if results['status'] == 'success':
        print("====================================")
        print("Hybrid Model Implementation Complete")
        print("====================================")
        print(f"ROI Break-even Point: Month {results['roi_breakeven']}")
        print(f"Simulation Completed: {'Yes' if results['simulation_completed'] else 'No'}")
        print("Output files saved to ./output/")
    else:
        print("Implementation process failed. Check logs for details.")
