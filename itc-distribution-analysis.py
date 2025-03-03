import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sqlite3
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set style for matplotlib visualizations
plt.style.use('ggplot')
sns.set_palette("Set2")

class DistributionAnalysis:
    """
    Class to analyze ITC distribution data comparing traditional system vs StackBox.
    """
    
    def __init__(self, data_path="./data/"):
        """
        Initialize the analysis class.
        
        Args:
            data_path: Path to the directory containing data files
        """
        self.data_path = data_path
        self.order_data = None
        self.satisfaction_data = None
        self.performance_data = None
        self.store_data = None
        self.financial_data = None
        
    def load_data(self):
        """
        Load data from various sources (CSV, Excel, SQL).
        """
        print("Loading datasets...")
        
        # Load order data from CSV
        try:
            self.order_data = pd.read_csv(os.path.join(self.data_path, "order_data.csv"))
            print("Order data loaded successfully.")
        except Exception as e:
            print(f"Error loading order data: {e}")
        
        # Load satisfaction data from Excel
        try:
            self.satisfaction_data = pd.read_excel(os.path.join(self.data_path, "customer_satisfaction.xlsx"))
            print("Satisfaction data loaded successfully.")
        except Exception as e:
            print(f"Error loading satisfaction data: {e}")
        
        # Load performance metrics from SQL
        try:
            conn = sqlite3.connect(os.path.join(self.data_path, "performance.db"))
            self.performance_data = pd.read_sql_query("SELECT * FROM performance_metrics", conn)
            conn.close()
            print("Performance data loaded successfully.")
        except Exception as e:
            print(f"Error loading performance data: {e}")
        
        # Load store information
        try:
            self.store_data = pd.read_csv(os.path.join(self.data_path, "store_info.csv"))
            print("Store data loaded successfully.")
        except Exception as e:
            print(f"Error loading store data: {e}")
        
        # Load financial data
        try:
            self.financial_data = pd.read_excel(os.path.join(self.data_path, "financial_metrics.xlsx"))
            print("Financial data loaded successfully.")
        except Exception as e:
            print(f"Error loading financial data: {e}")
        
        return self
    
    def clean_data(self):
        """
        Clean and preprocess the data.
        """
        print("Cleaning datasets...")
        
        # Handle missing values
        if self.order_data is not None:
            self.order_data = self.order_data.dropna(subset=['order_id', 'distribution_system'])
            self.order_data['order_date'] = pd.to_datetime(self.order_data['order_date'])
            
        if self.satisfaction_data is not None:
            self.satisfaction_data = self.satisfaction_data.fillna({
                'satisfaction_score': self.satisfaction_data['satisfaction_score'].median()
            })
            
        if self.performance_data is not None:
            # Convert date strings to datetime
            if 'date' in self.performance_data.columns:
                self.performance_data['date'] = pd.to_datetime(self.performance_data['date'])
            
        if self.store_data is not None:
            # Categorize store sizes
            size_bins = [0, 100, 300, 500, float('inf')]
            size_labels = ['Very Small', 'Small', 'Medium', 'Large']
            self.store_data['size_category'] = pd.cut(
                self.store_data['store_area_sqft'], 
                bins=size_bins, 
                labels=size_labels
            )
        
        return self
    
    def analyze_order_frequency(self):
        """
        Analyze order frequency between traditional system and StackBox.
        """
        if self.order_data is None:
            print("Order data not loaded. Please run load_data() first.")
            return None
        
        print("Analyzing order frequency...")
        
        # Group by month and distribution system
        self.order_data['month'] = self.order_data['order_date'].dt.strftime('%Y-%m')
        monthly_orders = self.order_data.groupby(['month', 'distribution_system']).size().unstack()
        
        # Calculate repeat orders by store
        store_order_counts = self.order_data.groupby(['store_id', 'distribution_system']).size().unstack()
        
        # Categorize stores by order frequency
        def categorize_frequency(count):
            if count <= 1:
                return 'One-time'
            elif count <= 3:
                return 'Occasional (1-3/month)'
            elif count <= 8:
                return 'Regular (4-8/month)'
            else:
                return 'Frequent (>8/month)'
        
        # Apply frequency categorization
        if 'Traditional' in store_order_counts.columns and 'StackBox' in store_order_counts.columns:
            trad_frequency = store_order_counts['Traditional'].dropna().apply(categorize_frequency).value_counts(normalize=True) * 100
            sb_frequency = store_order_counts['StackBox'].dropna().apply(categorize_frequency).value_counts(normalize=True) * 100
            
            frequency_df = pd.DataFrame({
                'Traditional': trad_frequency,
                'StackBox': sb_frequency
            }).fillna(0)
            
            return {
                'monthly_orders': monthly_orders,
                'frequency_distribution': frequency_df
            }
        else:
            print("Missing distribution system data in orders.")
            return None
    
    def analyze_satisfaction(self):
        """
        Analyze customer satisfaction between distribution systems.
        """
        if self.satisfaction_data is None:
            print("Satisfaction data not loaded. Please run load_data() first.")
            return None
        
        print("Analyzing customer satisfaction...")
        
        # Group satisfaction scores by category and distribution system
        def categorize_satisfaction(score):
            if score >= 4.5:
                return 'Very Satisfied'
            elif score >= 3.5:
                return 'Satisfied'
            elif score >= 2.5:
                return 'Neutral'
            elif score >= 1.5:
                return 'Dissatisfied'
            else:
                return 'Very Dissatisfied'
                
        self.satisfaction_data['satisfaction_category'] = self.satisfaction_data['satisfaction_score'].apply(categorize_satisfaction)
        
        # Calculate percentage distribution of satisfaction categories
        satisfaction_dist = pd.crosstab(
            self.satisfaction_data['satisfaction_category'], 
            self.satisfaction_data['distribution_system'],
            normalize='columns'
        ) * 100
        
        # Calculate average satisfaction by store size
        if self.store_data is not None:
            # Merge satisfaction data with store data
            merged_data = pd.merge(
                self.satisfaction_data, 
                self.store_data, 
                on='store_id'
            )
            
            # Calculate average satisfaction by store size and distribution system
            size_satisfaction = merged_data.groupby(['size_category', 'distribution_system'])['satisfaction_score'].mean().unstack()
            
            return {
                'satisfaction_distribution': satisfaction_dist,
                'size_satisfaction': size_satisfaction
            }
        else:
            return {
                'satisfaction_distribution': satisfaction_dist
            }
    
    def analyze_efficiency(self):
        """
        Analyze efficiency metrics between distribution systems.
        """
        if self.performance_data is None:
            print("Performance data not loaded. Please run load_data() first.")
            return None
        
        print("Analyzing efficiency metrics...")
        
        # Calculate average metrics by distribution system
        metrics = ['order_processing_time_mins', 'delivery_time_hours', 
                   'order_accuracy_percent', 'inventory_visibility_percent', 
                   'return_rate_percent']
        
        efficiency_metrics = self.performance_data.groupby('distribution_system')[metrics].mean()
        
        # Calculate processing time distribution
        def categorize_processing_time(time):
            if time < 15:
                return '<15'
            elif time < 30:
                return '15-30'
            elif time < 60:
                return '30-60'
            else:
                return '>60'
                
        if 'order_processing_time_mins' in self.performance_data.columns:
            self.performance_data['processing_time_category'] = self.performance_data['order_processing_time_mins'].apply(categorize_processing_time)
            
            processing_time_dist = pd.crosstab(
                self.performance_data['processing_time_category'], 
                self.performance_data['distribution_system'],
                normalize='columns'
            ) * 100
            
            return {
                'efficiency_metrics': efficiency_metrics,
                'processing_time_distribution': processing_time_dist
            }
        else:
            return {
                'efficiency_metrics': efficiency_metrics
            }
            
    def analyze_adoption_barriers(self):
        """
        Analyze barriers to StackBox adoption.
        """
        if self.store_data is None or 'adoption_barrier' not in self.store_data.columns:
            print("Store data with adoption barriers not available.")
            return None
            
        print("Analyzing adoption barriers...")
        
        # Count frequency of each barrier
        barriers = self.store_data[self.store_data['distribution_system'] == 'Traditional']['adoption_barrier'].value_counts(normalize=True) * 100
        
        # Analyze barriers by store size
        barriers_by_size = pd.crosstab(
            self.store_data['adoption_barrier'], 
            self.store_data['size_category'],
            normalize='columns'
        ) * 100
        
        return {
            'barriers': barriers,
            'barriers_by_size': barriers_by_size
        }
    
    def analyze_category_performance(self):
        """
        Analyze performance by product category.
        """
        if self.order_data is None or 'category' not in self.order_data.columns:
            print("Order data with product categories not available.")
            return None
            
        print("Analyzing category performance...")
        
        # Calculate total order value by category and distribution system
        category_value = self.order_data.groupby(['category', 'distribution_system'])['order_value'].sum().unstack()
        
        # Calculate order count by category
        category_count = self.order_data.groupby(['category', 'distribution_system']).size().unstack()
        
        # Normalize to create performance scores (0-100 scale)
        for col in category_value.columns:
            max_val = category_value[col].max()
            category_value[col] = (category_value[col] / max_val) * 100
            
        return {
            'category_value': category_value,
            'category_count': category_count
        }
    
    def analyze_cost_structure(self):
        """
        Analyze cost structure between distribution systems.
        """
        if self.financial_data is None:
            print("Financial data not loaded. Please run load_data() first.")
            return None
            
        print("Analyzing cost structure...")
        
        # Extract cost categories
        cost_data = self.financial_data[self.financial_data['metric_type'] == 'cost']
        
        # Calculate percentage of total cost for each category
        pivot_costs = cost_data.pivot_table(
            values='value', 
            index='category', 
            columns='distribution_system'
        )
        
        # Calculate percentage of total for each system
        for col in pivot_costs.columns:
            pivot_costs[col] = (pivot_costs[col] / pivot_costs[col].sum()) * 100
            
        return pivot_costs
        
    def generate_hybrid_model_projection(self):
        """
        Generate projection for hybrid model performance.
        """
        if self.order_data is None or self.financial_data is None:
            print("Order and financial data required for hybrid model projection.")
            return None
            
        print("Generating hybrid model projection...")
        
        # Calculate current monthly stats
        self.order_data['month'] = self.order_data['order_date'].dt.strftime('%Y-%m')
        monthly_stats = self.order_data.groupby(['month', 'distribution_system'])['order_value'].sum().unstack()
        
        # Get the last 6 months of data
        last_6_months = monthly_stats.tail(6)
        
        # Create projection for next 6 months
        # Hybrid model assumes: Dock delivery efficiency from StackBox, 
        # Last-mile satisfaction from Traditional
        projection_months = pd.date_range(
            start=pd.to_datetime(last_6_months.index[-1], format='%Y-%m') + pd.DateOffset(months=1), 
            periods=6, 
            freq='M'
        ).strftime('%Y-%m')
        
        # Traditional and StackBox projections (simple trend extrapolation)
        traditional_trend = np.polyfit(range(6), last_6_months['Traditional'].values, 1)
        stackbox_trend = np.polyfit(range(6), last_6_months['StackBox'].values, 1)
        
        traditional_projection = [traditional_trend[0] * (i + 6) + traditional_trend[1] for i in range(6)]
        stackbox_projection = [stackbox_trend[0] * (i + 6) + stackbox_trend[1] for i in range(6)]
        
        # Hybrid model projection (assumes 10-15% synergy benefit)
        hybrid_projection = []
        for i in range(6):
            base = max(traditional_projection[i], stackbox_projection[i])
            synergy = 0.1 + (i * 0.01)  # Increasing synergy over time
            hybrid_projection.append(base * (1 + synergy))
        
        projection_df = pd.DataFrame({
            'month': projection_months,
            'Traditional': traditional_projection,
            'StackBox': stackbox_projection,
            'Hybrid': hybrid_projection
        })
        
        projection_df.set_index('month', inplace=True)
        
        return projection_df
    
    def visualize_order_frequency(self, results, output_dir="./output/"):
        """
        Create visualizations for order frequency analysis.
        """
        if results is None:
            print("No results to visualize.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Monthly orders line chart
        plt.figure(figsize=(12, 6))
        results['monthly_orders'].plot(marker='o')
        plt.title('Monthly Order Volume: Traditional vs StackBox')
        plt.xlabel('Month')
        plt.ylabel('Number of Orders')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Distribution System')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_orders.png'))
        
        # Order frequency distribution
        plt.figure(figsize=(10, 6))
        results['frequency_distribution'].plot(kind='bar')
        plt.title('Order Frequency Distribution')
        plt.xlabel('Order Frequency Category')
        plt.ylabel('Percentage of Stores')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Distribution System')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'order_frequency.png'))
        
        # Interactive plotly version for dashboard
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Monthly Orders", "Order Frequency Distribution"),
            specs=[[{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Add monthly orders line traces
        for col in results['monthly_orders'].columns:
            fig.add_trace(
                go.Scatter(
                    x=results['monthly_orders'].index,
                    y=results['monthly_orders'][col],
                    name=col,
                    mode='lines+markers'
                ),
                row=1, col=1
            )
        
        # Add frequency distribution bar traces
        for col in results['frequency_distribution'].columns:
            fig.add_trace(
                go.Bar(
                    x=results['frequency_distribution'].index,
                    y=results['frequency_distribution'][col],
                    name=col
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            height=500,
            width=1000,
            title_text="Order Frequency Analysis"
        )
        
        fig.write_html(os.path.join(output_dir, 'order_frequency_interactive.html'))
        
    def visualize_satisfaction(self, results, output_dir="./output/"):
        """
        Create visualizations for satisfaction analysis.
        """
        if results is None:
            print("No results to visualize.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Satisfaction distribution
        plt.figure(figsize=(12, 6))
        results['satisfaction_distribution'].plot(kind='bar')
        plt.title('Customer Satisfaction Levels')
        plt.xlabel('Satisfaction Category')
        plt.ylabel('Percentage of Responses')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Distribution System')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'satisfaction_levels.png'))
        
        # Satisfaction by store size (if available)
        if 'size_satisfaction' in results:
            plt.figure(figsize=(10, 6))
            results['size_satisfaction'].plot(kind='bar')
            plt.title('Average Satisfaction by Store Size')
            plt.xlabel('Store Size Category')
            plt.ylabel('Average Satisfaction Score')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Distribution System')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'satisfaction_by_size.png'))
        
        # Interactive plotly version
        fig = go.Figure()
        
        for col in results['satisfaction_distribution'].columns:
            fig.add_trace(
                go.Bar(
                    x=results['satisfaction_distribution'].index,
                    y=results['satisfaction_distribution'][col],
                    name=col
                )
            )
        
        fig.update_layout(
            title='Customer Satisfaction Levels',
            xaxis_title='Satisfaction Category',
            yaxis_title='Percentage of Responses',
            barmode='group'
        )
        
        fig.write_html(os.path.join(output_dir, 'satisfaction_interactive.html'))
    
    def visualize_efficiency(self, results, output_dir="./output/"):
        """
        Create visualizations for efficiency metrics.
        """
        if results is None:
            print("No results to visualize.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Efficiency metrics bar chart
        plt.figure(figsize=(12, 6))
        results['efficiency_metrics'].plot(kind='bar')
        plt.title('Efficiency Metrics Comparison')
        plt.xlabel('Distribution System')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'efficiency_metrics.png'))
        
        # Processing time distribution (if available)
        if 'processing_time_distribution' in results:
            plt.figure(figsize=(10, 6))
            results['processing_time_distribution'].plot(kind='bar')
            plt.title('Order Processing Time Distribution')
            plt.xlabel('Processing Time (minutes)')
            plt.ylabel('Percentage of Orders')
            plt.grid(True, alpha=0.3)
            plt.legend(title='Distribution System')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'processing_time.png'))
        
        # Interactive plotly radar chart for efficiency metrics
        categories = results['efficiency_metrics'].columns.tolist()
        
        fig = go.Figure()
        
        for system in results['efficiency_metrics'].index:
            fig.add_trace(go.Scatterpolar(
                r=results['efficiency_metrics'].loc[system].values,
                theta=categories,
                fill='toself',
                name=system
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )
            ),
            showlegend=True,
            title='Efficiency Metrics Radar Chart'
        )
        
        fig.write_html(os.path.join(output_dir, 'efficiency_radar.html'))
    
    def visualize_adoption_barriers(self, results, output_dir="./output/"):
        """
        Create visualizations for adoption barriers.
        """
        if results is None:
            print("No results to visualize.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Adoption barriers pie chart
        plt.figure(figsize=(10, 8))
        results['barriers'].plot(kind='pie', autopct='%1.1f%%')
        plt.title('Barriers to StackBox Adoption')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'adoption_barriers.png'))
        
        # Barriers by store size heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(results['barriers_by_size'], annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title('Adoption Barriers by Store Size')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'barriers_by_size.png'))
        
        # Interactive plotly pie chart
        fig = go.Figure(data=[go.Pie(
            labels=results['barriers'].index,
            values=results['barriers'].values,
            hole=.3,
            textinfo='label+percent'
        )])
        
        fig.update_layout(title_text='Barriers to StackBox Adoption')
        fig.write_html(os.path.join(output_dir, 'adoption_barriers_interactive.html'))
    
    def visualize_category_performance(self, results, output_dir="./output/"):
        """
        Create visualizations for category performance.
        """
        if results is None:
            print("No results to visualize.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Category performance radar chart
        categories = results['category_value'].index.tolist()
        traditional_values = results['category_value']['Traditional'].values.tolist()
        stackbox_values = results['category_value']['StackBox'].values.tolist()
        
        # Close the loop for radar chart
        categories.append(categories[0])
        traditional_values.append(traditional_values[0])
        stackbox_values.append(stackbox_values[0])
        
        plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, polar=True)
        
        # Plot Traditional
        ax.plot(np.linspace(0, 2*np.pi, len(categories)), traditional_values, 'o-', linewidth=2, label='Traditional')
        ax.fill(np.linspace(0, 2*np.pi, len(categories)), traditional_values, alpha=0.25)
        
        # Plot StackBox
        ax.plot(np.linspace(0, 2*np.pi, len(categories)), stackbox_values, 'o-', linewidth=2, label='StackBox')
        ax.fill(np.linspace(0, 2*np.pi, len(categories)), stackbox_values, alpha=0.25)
        
        ax.set_thetagrids(np.degrees(np.linspace(0, 2*np.pi, len(categories)-1)), categories[:-1])
        plt.title('Category Performance Radar')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'category_radar.png'))
        
        # Interactive plotly radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=results['category_value']['Traditional'].values,
            theta=results['category_value'].index,
            fill='toself',
            name='Traditional'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=results['category_value']['StackBox'].values,
            theta=results['category_value'].index,
            fill='toself',
            name='StackBox'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title='Category Performance Comparison'
        )
        
        fig.write_html(os.path.join(output_dir, 'category_radar_interactive.html'))
    
    def visualize_hybrid_projection(self, projection, output_dir="./output/"):
        """
        Create visualizations for hybrid model projection.
        """
        if projection is None:
            print("No projection data to visualize.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Line chart for projection
        plt.figure(figsize=(12, 6))
        projection.plot(marker='o')
        plt.title('Projected Order Value by Distribution Model')
        plt.xlabel('Month')
        plt.ylabel('Order Value')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Distribution System')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hybrid_projection.png'))
        
        # Interactive plotly version
        fig = go.Figure()
        
        for col in projection.columns:
            fig.add_trace(
                go.Scatter(
                    x=projection.index,
                    y=projection[col],
                    mode='lines+markers',
                    name=col
                )
            )
        
        fig.update_layout(
            title='Projected Order Value by Distribution Model',
            xaxis_title='Month',
            yaxis_title='Order Value',
            legend_title='Distribution System'
        )
        
        fig.write_html(os.path.join(output_dir, 'hybrid_projection_interactive.html'))
    
    def create_dashboard(self, output_dir="./output/"):
        """
        Create a comprehensive dashboard of all visualizations.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all interactive HTML visualizations
        html_files = [f for f in os.listdir(output_dir) if f.endswith('_interactive.html')]
        
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ITC Distribution Analysis Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .header { background-color: #4a4a4a; color: white; padding: 20px; text-align: center; margin-bottom: 20px; }
                .dashboard-container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
                .dashboard-item { background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: calc(50% - 20px); height: 500px; overflow: hidden; }
                .dashboard-item h3 { margin: 0; padding: 15px; background-color: #f0f0f0; border-bottom: 1px solid #ddd; }
                .dashboard-item iframe { width: 100%; height: calc(100% - 50px); border: none; }
                .footer { margin-top: 30px; text-align: center; color: #666; }
                @media (max-width: 1100px) {
                    .dashboard-item { width: 100%; }
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ITC Distribution Model Analysis</h1>
                <p>Comparison of Traditional System vs StackBox Technology-Driven Distribution</p>
            </div>
            
            <div class="dashboard-container">
        """
        
        # Add each visualization to the dashboard
        for html_file in html_files:
            title = ' '.join(html_file.replace('_interactive.html', '').split('_')).title()
            dashboard_html += f"""
                <div class="dashboard-item">
                    <h3>{title}</h3>
                    <iframe src="{html_file}" frameborder="0"></iframe>
                </div>
            """
        
        dashboard_html += """
            </div>
            
            <div class="footer">
                <p>ITC Distribution Analysis Dashboard - Generated on {}</p>
            </div>
        </body>
        </html>
        """.format(datetime.now().strftime("%Y-%m-%d"))
        
        with open(os.path.join(output_dir, 'dashboard.html'), 'w') as f:
            f.write(dashboard_html)
        
        print(f"Dashboard created at {os.path.join(output_dir, 'dashboard.html')}")
    
    def run_full_analysis(self, output_dir="./output/"):
        """
        Run the complete analysis pipeline and generate all visualizations.
        """
        self.load_data()
        self.clean_data()
        
        # Run analyses
        order_results = self.analyze_order_frequency()
        satisfaction_results = self.analyze_satisfaction()
        efficiency_results = self.analyze_efficiency()
        barrier_results = self.analyze_adoption_barriers()
        category_results = self.analyze_category_performance()
        cost_structure = self.analyze_cost_structure()
        hybrid_projection = self.generate_hybrid_model_projection()
        
        # Generate visualizations
        self.visualize_order_frequency(order_results, output_dir)
        self.visualize_satisfaction(satisfaction_results, output_dir)
        self.visualize_efficiency(efficiency_results, output_dir)
        self.visualize_adoption_barriers(barrier_results, output_dir)
        self.visualize_category_performance(category_results, output_dir)
        self.visualize_hybrid_projection(hybrid_projection, output_dir)
        
        # Create comprehensive dashboard
        self.create_dashboard(output_dir)
        
        print("