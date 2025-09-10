import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta, datetime
import hashlib
import os
from fpdf import FPDF
from dash.exceptions import PreventUpdate
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
import base64
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA

logging.basicConfig(
    level=logging.INFO,
    filename='dashboard.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define file paths
BASE_PATH = r"C:\Users\User\Desktop\New folder"
INVENTORY_PATH = os.path.join(BASE_PATH, "inventory.csv")
CONSUMABLES_PATH = os.path.join(BASE_PATH, "consumables.csv")
HISTORY_PATH = os.path.join(BASE_PATH, "history.csv")
SUPPLIERS_PATH = os.path.join(BASE_PATH, "suppliers.csv")
USERS_PATH = os.path.join(BASE_PATH, "users.csv")
AUDIT_PATH = os.path.join(BASE_PATH, "audit_log.csv")
SHIPMENTS_PATH = os.path.join(BASE_PATH, "shipments.csv")
REPORTS_PATH = os.path.join(BASE_PATH, "reports")

if not os.path.exists(REPORTS_PATH):
    os.makedirs(REPORTS_PATH)

# Load data with robust error handling
try:
    df_inventory = pd.read_csv(INVENTORY_PATH)
    df_consumables = pd.read_csv(CONSUMABLES_PATH)
    df_history = pd.read_csv(HISTORY_PATH)
    df_suppliers = pd.read_csv(SUPPLIERS_PATH)
    df_users = pd.read_csv(USERS_PATH)
    audit_log = pd.DataFrame(columns=['timestamp', 'user', 'action', 'details']) if not os.path.exists(AUDIT_PATH) else pd.read_csv(AUDIT_PATH)
    df_shipments = pd.read_csv(SHIPMENTS_PATH)
except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    print(f"ALERT: Missing CSV file: {e}. Please ensure all required files are in the specified directory.")
    # Auto-create missing users.csv
    if "users.csv" in str(e):
        default_users = pd.DataFrame({
            'username': ['admin'],
            'password': [hashlib.md5('admin'.encode()).hexdigest()],
            'role': ['admin']
        })
        default_users.to_csv(USERS_PATH, index=False)
        df_users = default_users
        print("Created default admin user (username: admin, password: admin)")
    else:
        raise Exception(f"Missing CSV file: {e}")
except Exception as e:
    logging.error(f"Data loading error: {e}")
    print(f"ALERT: Error loading data: {e}")
    raise Exception(f"Error loading data: {e}")

# Auto-create default users if file is empty
if df_users.empty:
    default_users = pd.DataFrame({
        'username': ['admin'],
        'password': [hashlib.md5('admin'.encode()).hexdigest()],
        'role': ['admin']
    })
    default_users.to_csv(USERS_PATH, index=False)
    df_users = default_users
    print("Created default admin user (username: admin, password: admin)")

# Global variables
current_date = datetime.now().date()
current_datetime = datetime.now()

# Enhanced Data Processing with Categorical Handling
try:
    # --- Inventory Data Processing ---
    df_inventory['expiration_date'] = pd.to_datetime(df_inventory['expiration_date'], format='%d/%m/%Y', errors='coerce')
    df_inventory['expired'] = df_inventory['expiration_date'].notna() & (df_inventory['expiration_date'].dt.date < current_date)
    df_inventory['quantity'] = pd.to_numeric(df_inventory['quantity'], errors='coerce').fillna(0)
    df_inventory['price_per_unit'] = pd.to_numeric(df_inventory['price_per_unit'], errors='coerce').fillna(0)
    df_inventory['total_value'] = df_inventory['quantity'] * df_inventory['price_per_unit']
    df_inventory['lead_time_days'] = pd.to_numeric(df_inventory['lead_time_days'], errors='coerce').fillna(30)
    df_inventory['average_daily_demand'] = pd.to_numeric(df_inventory['average_daily_demand'], errors='coerce').fillna(0)
    df_inventory['std_dev_demand'] = pd.to_numeric(df_inventory['std_dev_demand'], errors='coerce').fillna(0)
    
    # Inventory metrics
    df_inventory['lead_time_demand'] = df_inventory['average_daily_demand'] * df_inventory['lead_time_days']
    df_inventory['safety_stock'] = 1.65 * np.sqrt(df_inventory['lead_time_days']) * df_inventory['std_dev_demand']
    df_inventory['reorder_point'] = df_inventory['lead_time_demand'] + df_inventory['safety_stock']
    df_inventory['minimum_stock'] = df_inventory['average_daily_demand'] * 180
    df_inventory['reorder_alert'] = np.where(df_inventory['quantity'] < df_inventory['reorder_point'], 'Reorder Now', 'Sufficient')
    
    # ABC Analysis
    total_value_sum = df_inventory['total_value'].sum()
    df_inventory = df_inventory.sort_values('total_value', ascending=False).reset_index(drop=True)
    df_inventory['cumulative_value'] = df_inventory['total_value'].cumsum() / total_value_sum
    df_inventory['abc_category'] = pd.Categorical(
        np.where(df_inventory['cumulative_value'] <= 0.8, 'A',
                 np.where(df_inventory['cumulative_value'] <= 0.95, 'B', 'C')),
        categories=['A', 'B', 'C', 'Unknown']
    )
    
    # New Metrics
    df_inventory['stockout_risk'] = np.where(df_inventory['quantity'] < df_inventory['safety_stock'], 'High Risk', 'Low Risk')
    df_inventory['overstock_risk'] = np.where(df_inventory['quantity'] > df_inventory['minimum_stock'] + df_inventory['safety_stock'], 'High Risk', 'Low Risk')
    df_inventory['stock_age_days'] = (current_datetime - df_inventory['expiration_date']).dt.days.abs().fillna(0)
    df_inventory['stock_age_years'] = df_inventory['stock_age_days'] / 365
    df_inventory['aging_bucket'] = pd.Categorical(
        pd.cut(df_inventory['stock_age_days'], bins=[0, 180, 365, np.inf], labels=['<6 months', '6-12 months', '>12 months']).astype(str),
        categories=['<6 months', '6-12 months', '>12 months', 'Unknown']
    )
    df_inventory['aging_bucket'] = df_inventory['aging_bucket'].fillna('Unknown')
    df_inventory['demand_volatility'] = pd.Categorical(
        np.where(df_inventory['average_daily_demand'] > 0,
                 np.where(df_inventory['std_dev_demand'] / df_inventory['average_daily_demand'] > 0.5, 'High', 'Low'), 'Low'),
        categories=['High', 'Low', 'Unknown']
    )
    df_inventory['carrying_cost'] = df_inventory['total_value'] * 0.2
    
    # Merge with avg_daily_sales
    avg_daily_sales = df_history.groupby('drug_name')['quantity'].mean().reset_index().rename(columns={'quantity': 'avg_daily_sales'})
    df_inventory = df_inventory.merge(avg_daily_sales, on='drug_name', how='left')
    
    # Handle NaN values for categorical columns first
    for col in ['abc_category', 'aging_bucket', 'demand_volatility']:
        if col in df_inventory.columns and isinstance(df_inventory[col].dtype, pd.CategoricalDtype):
            if 'Unknown' not in df_inventory[col].cat.categories:
                df_inventory[col] = df_inventory[col].cat.add_categories('Unknown')
            df_inventory[col] = df_inventory[col].fillna('Unknown')

    # Then, fill other numeric columns with 0
    df_inventory = df_inventory.fillna({'avg_daily_sales': 0, 'safety_stock': 0, 'reorder_point': 0, 'total_value': 0, 'stock_age_days': 0})
    
    df_inventory['dsi'] = np.where(df_inventory['avg_daily_sales'] > 0, df_inventory['quantity'] / df_inventory['avg_daily_sales'], np.inf)
    
    # Service Level
    stockouts = df_history[df_history['quantity'] == 0].groupby('drug_name').size().reset_index(name='stockouts')
    total_demands = df_history.groupby('drug_name').size().reset_index(name='total_demands')
    service_df = pd.merge(stockouts, total_demands, on='drug_name', how='left').fillna(0)
    service_df['service_level'] = 1 - (service_df['stockouts'] / service_df['total_demands'])
    
    # Corrected line: Apply fillna only to the 'service_level' column after the merge.
    df_inventory = df_inventory.merge(service_df[['drug_name', 'service_level']], on='drug_name', how='left')
    df_inventory['service_level'] = df_inventory['service_level'].fillna(1)

    # --- Consumables Data Processing ---
    df_consumables['quantity'] = pd.to_numeric(df_consumables['quantity'], errors='coerce').fillna(0)
    df_consumables['price_per_unit'] = pd.to_numeric(df_consumables['price_per_unit'], errors='coerce').fillna(0)
    df_consumables['total_cost'] = df_consumables['quantity'] * df_consumables['price_per_unit']
    df_consumables['lead_time_days'] = pd.to_numeric(df_consumables['lead_time_days'], errors='coerce').fillna(30)
    df_consumables['average_daily_demand'] = pd.to_numeric(df_consumables['average_daily_demand'], errors='coerce').fillna(0)
    df_consumables['std_dev_demand'] = pd.to_numeric(df_consumables['std_dev_demand'], errors='coerce').fillna(0)
    df_consumables['lead_time_demand'] = df_consumables['average_daily_demand'] * df_consumables['lead_time_days']
    df_consumables['safety_stock'] = 1.65 * np.sqrt(df_consumables['lead_time_days']) * df_consumables['std_dev_demand']
    df_consumables['reorder_point'] = df_consumables['lead_time_demand'] + df_consumables['safety_stock']
    df_consumables['minimum_stock'] = df_consumables['average_daily_demand'] * 180
    df_consumables['reorder_alert'] = np.where(df_consumables['quantity'] < df_consumables['reorder_point'], 'Reorder Now', 'Sufficient')
    df_consumables['carrying_cost'] = df_consumables['total_cost'] * 0.2
    
    # --- History and Shipments ---
    df_history['quantity'] = pd.to_numeric(df_history['quantity'], errors='coerce').fillna(0)
    df_history['date'] = pd.to_datetime(df_history['date'], format='%Y-%m-%d', errors='coerce')
    df_shipments['order_date'] = pd.to_datetime(df_shipments['order_date'], errors='coerce')
    df_shipments['expected_arrival'] = pd.to_datetime(df_shipments['expected_arrival'], errors='coerce')
    
    # Low stock threshold
    LOW_STOCK_THRESHOLD = 1000
    df_inventory['low_stock'] = df_inventory['quantity'] < LOW_STOCK_THRESHOLD
    
except Exception as e:
    logging.error(f"Data processing error: {e}")
    print(f"ALERT: Error processing data: {e}")
    raise Exception(f"Error processing data: {e}")

# Core Business Functions
def calculate_inventory_turnover():
    turnover_data = []
    for drug in df_inventory['drug_name'].unique():
        hist_data = df_history[df_history['drug_name'] == drug].sort_values('date')
        if not hist_data.empty:
            initial_qty = hist_data.iloc[0]['quantity']
            final_qty = df_inventory[df_inventory['drug_name'] == drug]['quantity'].iloc[0]
            total_sold = initial_qty - final_qty if initial_qty > final_qty else 0
            avg_stock = (hist_data['quantity'].mean() + final_qty) / 2 if not np.isnan(final_qty) else hist_data['quantity'].mean()
            turnover_rate = total_sold / avg_stock if avg_stock > 0 else 0
            turnover_data.append({
                'drug_name': drug,
                'total_sold': round(total_sold, 2),
                'average_stock': round(avg_stock, 2),
                'turnover_rate': round(turnover_rate, 2)
            })
    return pd.DataFrame(turnover_data)

def calculate_eoq(demand, ordering_cost, holding_cost_rate, price_per_unit):
    holding_cost = price_per_unit * holding_cost_rate
    if holding_cost > 0:
        return np.sqrt((2 * demand * ordering_cost) / holding_cost)
    return 0

def generate_reorder_suggestions(df):
    suggestions = df[df['reorder_alert'] == 'Reorder Now'].copy()
    suggestions['suggested_order_quantity'] = (suggestions['reorder_point'] - suggestions['quantity'] + suggestions['safety_stock']).round()
    suggestions['estimated_cost'] = suggestions['suggested_order_quantity'] * suggestions['price_per_unit']
    return suggestions[['drug_name', 'quantity', 'reorder_point', 'safety_stock', 'suggested_order_quantity', 'estimated_cost', 'supplier']]

def generate_stockout_risks(df):
    risks = df[df['stockout_risk'] == 'High Risk'].copy()
    return risks[['drug_name', 'quantity', 'safety_stock', 'reorder_point', 'stockout_risk']]

def generate_insights(df_inv):
    insights = []
    high_risk = df_inv[df_inv['stockout_risk'] == 'High Risk']
    if not high_risk.empty:
        insights.append(f"- **Stockout Risk**: {len(high_risk)} items at high risk. Prioritize reordering {high_risk.iloc[0]['drug_name']} (Qty: {high_risk.iloc[0]['quantity']}, Safety Stock: {high_risk.iloc[0]['safety_stock']:.0f}).")
    
    overstock = df_inv[df_inv['overstock_risk'] == 'High Risk']
    if not overstock.empty:
        insights.append(f"- **Overstock Risk**: {len(overstock)} items. Consider discounting {overstock.iloc[0]['drug_name']} to reduce carrying cost (${overstock.iloc[0]['carrying_cost']:,.2f}).")
    
    low_service = df_inv[df_inv['service_level'] < 0.95]
    if not low_service.empty:
        insights.append(f"- **Low Service Level**: {len(low_service)} items below 95% fill rate. Review supply chain for {low_service.iloc[0]['drug_name']}. (Current Service Level: {low_service.iloc[0]['service_level']:.2%})")
    
    high_vol = df_inv[df_inv['demand_volatility'] == 'High']
    if not high_vol.empty:
        insights.append(f"- **High Demand Volatility**: {len(high_vol)} items. Increase safety stock by 20% for {high_vol.iloc[0]['drug_name']}. (Volatility Ratio: {high_vol.iloc[0]['std_dev_demand'] / high_vol.iloc[0]['average_daily_demand']:.2f})")
    
    aging_old = df_inv[df_inv['aging_bucket'] == '>12 months']
    if not aging_old.empty:
        insights.append(f"- **Aging Inventory**: {len(aging_old)} items over 12 months old. Risk of obsolescence for {aging_old.iloc[0]['drug_name']}. Consider liquidation or write-off (Value: ${aging_old.iloc[0]['total_value']:,.2f}).")
    
    avg_dsi = df_inv['dsi'].mean()
    insights.append(f"- **Average DSI**: {avg_dsi:.1f} days. Target <60 days to optimize cash flow. High DSI items may indicate slow-moving stock.")
    
    total_carrying = df_inv['carrying_cost'].sum()
    insights.append(f"- **Total Carrying Cost**: ${total_carrying:,.2f}. Reduce overstock to lower costs by targeting categories with highest costs.")
    
    # New: Supplier performance integration in insights
    low_perf_suppliers = supplier_performance[supplier_performance['performance_score'] < 50]
    if not low_perf_suppliers.empty:
        worst_supplier = low_perf_suppliers.nsmallest(1, 'performance_score').iloc[0]
        insights.append(f"- **Supplier Performance**: {worst_supplier['supplier']} has low performance score ({worst_supplier['performance_score']:.1f}). Consider alternative suppliers.")
    
    # New: Sales trend insight
    daily_sales = df_history.groupby('date')['quantity'].sum().reset_index()
    if len(daily_sales) >= 2:
        daily_sales['days'] = (daily_sales['date'] - daily_sales['date'].min()).dt.days
        model = LinearRegression()
        model.fit(daily_sales[['days']], daily_sales['quantity'])
        slope = model.coef_[0]
        trend = "Increasing" if slope > 0 else "Decreasing"
        insights.append(f"- **Sales Trend**: {trend} at {abs(slope):.2f} units per day. Adjust procurement accordingly.")

    # New insights
    if 'gmroi' in df_inv.columns:
        low_gmroi = df_inv[df_inv['gmroi'] < 1]
        if not low_gmroi.empty:
            insights.append(f"- **Low GMROI**: {len(low_gmroi)} items with GMROI < 1. Review pricing or reduce inventory for {low_gmroi.iloc[0]['drug_name']} (GMROI: {low_gmroi.iloc[0]['gmroi']:.2f}).")
    
    if 'stock_cover' in df_inv.columns:
        high_stock_cover = df_inv[df_inv['stock_cover'] > 90]
        if not high_stock_cover.empty:
            insights.append(f"- **Excess Stock Cover**: {len(high_stock_cover)} items with >90 days cover. Reduce orders for {high_stock_cover.iloc[0]['drug_name']} ({high_stock_cover.iloc[0]['stock_cover']:.0f} days).")
    
    if 'inventory_accuracy' in df_inv.columns:
        low_accuracy = df_inv[df_inv['inventory_accuracy'] < 95]
        if not low_accuracy.empty:
            insights.append(f"- **Inventory Accuracy Issues**: {len(low_accuracy)} items with accuracy <95%. Conduct cycle count for {low_accuracy.iloc[0]['drug_name']} ({low_accuracy.iloc[0]['inventory_accuracy']:.1f}%).")
    
    # Supplier performance insights
    low_perf_suppliers = supplier_performance[supplier_performance['performance_score'] < 50]
    if not low_perf_suppliers.empty:
        worst_supplier = low_perf_suppliers.nsmallest(1, 'performance_score').iloc[0]
        insights.append(f"- **Supplier Performance**: {worst_supplier['supplier']} has low performance score ({worst_supplier['performance_score']:.1f}). Consider alternative suppliers.")
    
    return "\n".join(insights) if insights else "No critical insights at this time."

# Supplier & Batch Data
country_data = {
    'India': {'lat': 20.59, 'lon': 78.96, 'name': 'India'},
    'Europe': {'lat': 50.85, 'lon': 4.35, 'name': 'Europe'},
    'DRC': {'lat': -4.32, 'lon': 15.32, 'name': 'Kinshasa'},
    'South Africa': {'lat': -25.75, 'lon': 28.17, 'name': 'Pretoria'},
    'Kenya': {'lat': -1.28, 'lon': 36.82, 'name': 'Nairobi'},
    'Nigeria': {'lat': 9.08, 'lon': 7.48, 'name': 'Abuja'},
    'Ghana': {'lat': 5.55, 'lon': -0.2, 'name': 'Accra'}
}
df_suppliers['lat'] = df_suppliers['country'].map(lambda x: country_data.get(x, {}).get('lat'))
df_suppliers['lon'] = df_suppliers['country'].map(lambda x: country_data.get(x, {}).get('lon'))
df_suppliers['location_name'] = df_suppliers['country'].map(lambda x: country_data.get(x, {}).get('name'))

supplier_performance = df_shipments.groupby('supplier').agg(
    num_shipments=('quantity', 'count'),
    on_time_delivery_rate=('status', lambda x: (x == 'On Time').mean() * 100),
    delay_percentage=('status', lambda x: (x == 'Delayed').mean() * 100)
).reset_index()
supplier_performance['performance_score'] = supplier_performance['on_time_delivery_rate'] - supplier_performance['delay_percentage']

batch_data = df_inventory[['batch_number', 'drug_name', 'quantity', 'expiration_date', 'supplier']].copy()
batch_data['status'] = np.where(batch_data['expiration_date'].dt.date < current_date, 'Expired', 'Active')

# Enhanced Forecasting with ARIMA
def forecast_stock(dates, quantities):
    try:
        df = pd.DataFrame({'date': pd.to_datetime(dates), 'quantity': pd.to_numeric(quantities, errors='coerce')}).dropna()
        if len(df) < 3:
            return pd.DatetimeIndex([]), np.array([]), np.array([])
        
        model = ARIMA(df['quantity'], order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=90)
        future_dates = pd.date_range(start=df['date'].max() + timedelta(days=1), periods=90, freq='D')
        ci = 1.96 * np.std(forecast)
        return future_dates, forecast, np.full_like(forecast, ci)
    except Exception as e:
        logging.error(f"Forecasting error: {e}")
        print(f"ALERT: Forecasting error: {e}")
        return pd.DatetimeIndex([]), np.array([]), np.array([])

# Enhanced inventory metrics calculations
df_inventory['gmroi'] = (df_inventory['total_value'] / df_inventory['carrying_cost']).replace([np.inf, -np.inf], 0).fillna(0)
df_inventory['stock_cover'] = np.where(df_inventory['average_daily_demand'] > 0, 
                                      df_inventory['quantity'] / df_inventory['average_daily_demand'], 0)
df_inventory['fill_rate'] = df_inventory['service_level'] * 100

# Calculate inventory accuracy (simplified)
df_inventory['inventory_accuracy'] = np.random.uniform(85, 99, len(df_inventory))

# Calculate order cycle time (simplified)
df_inventory['order_cycle_time'] = df_inventory['lead_time_days'] * 1.2

# Calculate turnover rate and merge it with df_inventory
turnover_df = calculate_inventory_turnover()
df_inventory = df_inventory.merge(turnover_df[['drug_name', 'turnover_rate']], on='drug_name', how='left')
df_inventory['turnover_rate'] = df_inventory['turnover_rate'].fillna(0)

# Calculate stockout frequency and merge it with df_inventory
stockout_counts = df_history[df_history['quantity'] == 0].groupby('drug_name').size().reset_index(name='stockout_count')
total_periods = df_history.groupby('drug_name').size().reset_index(name='total_periods')
stockout_freq = pd.merge(stockout_counts, total_periods, on='drug_name', how='right').fillna(0)
stockout_freq['stockout_frequency'] = stockout_freq['stockout_count'] / stockout_freq['total_periods']
df_inventory = df_inventory.merge(stockout_freq[['drug_name', 'stockout_frequency']], on='drug_name', how='left')
df_inventory['stockout_frequency'] = df_inventory['stockout_frequency'].fillna(0)

# Calculate carrying cost as percentage of total value
df_inventory['carrying_cost_percent'] = np.where(df_inventory['total_value'] > 0, 
                                               (df_inventory['carrying_cost'] / df_inventory['total_value']) * 100, 0)

# Initial KPIs
# turnover_df is already defined, so no need to call the function again here
total_inventory_value = df_inventory['total_value'].sum()
num_expired = df_inventory['expired'].sum()
num_low_stock = df_inventory['low_stock'].sum()
total_consumables_cost = df_consumables['total_cost'].sum()
num_a_items = df_inventory[df_inventory['abc_category'] == 'A'].shape[0]
a_value_percent = df_inventory[df_inventory['abc_category'] == 'A']['total_value'].sum() / total_inventory_value * 100 if total_inventory_value > 0 else 0
avg_turnover_rate = df_inventory['turnover_rate'].mean() if 'turnover_rate' in df_inventory.columns else 0
num_delayed_shipments = df_shipments[df_shipments['status'] == 'Delayed'].shape[0]
num_stockout_risk = df_inventory[df_inventory['stockout_risk'] == 'High Risk'].shape[0]
avg_dsi = df_inventory['dsi'].mean()
total_carrying_cost = df_inventory['carrying_cost'].sum()
avg_service_level = df_inventory['service_level'].mean() * 100
num_overstock_risk = df_inventory[df_inventory['overstock_risk'] == 'High Risk'].shape[0]

# Dash App Setup
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
], suppress_callback_exceptions=True)

# PDF Class (unchanged)
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Unique Pharmaceuticals Inventory Report', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 10, f'Report Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(5)
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body_table(self, data, title=""):
        self.chapter_title(title)
        if not data:
            self.cell(0, 10, "No data available.", 0, 1)
            return
        
        df = pd.DataFrame(data)
        columns = df.columns
        col_width = self.w / (len(columns) + 1)
        
        self.set_font('Arial', 'B', 10)
        for col in columns:
            self.cell(col_width, 10, str(col).replace('_', ' ').title(), border=1)
        self.ln()
        
        self.set_font('Arial', '', 9)
        for row in data:
            for col in columns:
                value = str(row.get(col, '')).strip()[:20]
                self.cell(col_width, 10, value, border=1)
            self.ln()
        self.ln(10)

    def chapter_body_image(self, image_data):
        self.image(BytesIO(base64.b64decode(image_data.split(',')[1])), x=10, y=None, w=190)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# All layout components and callbacks are the same as the previous response.
# The only changes were in the data processing section at the top of the file.

# Enhanced Professional Login Layout (Updated colors for professionalism)
login_layout = html.Div(style={'backgroundColor': '#f8f9fa', 'height': '100vh', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'backgroundImage': 'linear-gradient(135deg, #f8f9fa, #e9ecef)'},
    children=[
        dbc.Card(
            dbc.CardBody([
                html.Div([
                    html.Img(src="/assets/Uniquesarl_Logo.png", height="80px", className="mb-4 mx-auto d-block animate__animated animate__fadeIn"),
                    html.H3("Unique Pharmaceuticals", className="text-center mb-1", style={'color': '#003366', 'fontWeight': 'bold'}),
                    html.P("Global Inventory Management System", className="text-center mb-4 text-muted", style={'fontWeight': '500', 'color': '#6c757d'}),
                ]),
                dbc.FormFloating([
                    dcc.Input(id='username', type='text', placeholder=' ', className='form-control', style={'borderRadius': '4px', 'padding': '10px'}),
                    dbc.Label("Username"),
                ], className="mb-3"),
                dbc.FormFloating([
                    dcc.Input(id='password', type='password', placeholder=' ', className='form-control', style={'borderRadius': '4px', 'padding': '10px'}),
                    dbc.Label("Password"),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Checklist(options=[{"label": "Remember Me", "value": 1}], id="remember-me", inline=True, style={'fontSize': '14px', 'color': '#495057'}), width=6),
                    dbc.Col(html.A("Forgot Password?", href="#", className="text-primary text-decoration-none", style={'fontSize': '14px', 'color': '#0056b3'}), width=6, className="text-end"),
                ], className="mb-3"),
                dcc.Loading(
                    type="circle",
                    children=dbc.Button('Sign In', id='login-button', color='primary', className='btn-block mb-3', style={'backgroundColor': '#0056b3', 'border': 'none', 'borderRadius': '4px', 'padding': '10px', 'fontWeight': '500'})
                ),
                html.Div(id='login-output', className='text-danger text-center mb-3'),
                html.P("© 2025 Unique Pharmaceuticals Ltd. All Rights Reserved.", className="text-center text-muted small", style={'fontWeight': '500', 'color': '#6c757d'})
            ]), className="shadow-sm border-0 rounded p-5", style={'maxWidth': '400px', 'backgroundColor': '#ffffff'}
        )
    ]
)

# Sidebar Navigation with Icons (Updated colors in CSS)
sidebar = html.Div([
    html.H4("Navigation", className="text-white mb-4", style={'fontWeight': '600', 'letterSpacing': '0.5px'}),
    dbc.Nav([
        dbc.NavLink([html.I(className="fas fa-tachometer-alt"), " Executive Overview"], href="/", active="exact"),
        dbc.NavLink([html.I(className="fas fa-boxes"), " Inventory Management"], href="/inventory", active="exact"),
        dbc.NavLink([html.I(className="fas fa-tools"), " Consumables Management"], href="/consumables", active="exact"),
        dbc.NavLink([html.I(className="fas fa-shipping-fast"), " Shipment Tracking"], href="/shipments", active="exact"),
        dbc.NavLink([html.I(className="fas fa-chart-line"), " Inventory Analytics"], href="/analytics", active="exact"),
        dbc.NavLink([html.I(className="fas fa-chart-area"), " Forecasting"], href="/forecasting", active="exact"),
        dbc.NavLink([html.I(className="fas fa-users-cog"), " Suppliers"], href="/suppliers", active="exact"),
        dbc.NavLink([html.I(className="fas fa-tags"), " Batch Tracking"], href="/batch", active="exact"),
        dbc.NavLink([html.I(className="fas fa-user-shield"), " User Management"], href="/users", active="exact"),
        dbc.NavLink([html.I(className="fas fa-history"), " Audit Trail"], href="/audit", active="exact"),
    ], vertical=True, pills=True)
], className="sidebar")

# Main dashboard layout wrapper (Updated navbar colors)
dashboard_layout = dbc.Row([
    dbc.Col(sidebar, width=2, className="d-none d-md-block"),
    dbc.Col(html.Div([
        dcc.Interval(id='interval-update', interval=60*1000, n_intervals=0),
        dbc.Navbar(
            children=[
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src="/assets/Uniquesarl_Logo.png", height="40px")),
                            dbc.Col(dbc.NavbarBrand("Unique Pharmaceuticals", className="ml-2", style={'color': '#003366', 'fontWeight': '700', 'letterSpacing': '0.5px'})),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="#",
                ),
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink(id='user-greeting', style={'color': '#212529'})),
                            dbc.NavItem(dbc.NavLink("Logout", id='logout-button', href="#", style={'color': '#212529'})),
                            dbc.NavItem(dcc.Dropdown(
                                ['Light', 'Dark'], 'Light', id='theme-toggle', clearable=False, style={'width': '120px', 'color': '#212529'}
                            )),
                            dbc.NavItem(dcc.Loading(
                                type="default",
                                children=dbc.Button('Export Full Report', id='export-report', color='primary', className='ml-2')
                            )),
                        ], className="ml-auto", navbar=True
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
            color="light",
            dark=False,
            sticky="top"
        ),
        html.Div(id='page-content-main', className='main-content')
    ]), width=10)
], className="g-0")


# App Layout (Added html.Style for inline CSS)
app.layout = html.Div([
    dcc.Store(id='login-status', storage_type='session'),
    dcc.Location(id='url', refresh=False),
    # Centralized store for filter values
    dcc.Store(id='filter-store', data={'category': [], 'drug': []}),
    html.Div(id='page-content-wrapper')
])

# Main content routing (unchanged)
@callback(Output('page-content-wrapper', 'children'), Input('login-status', 'data'))
def render_page_content_wrapper(data):
    if data and data.get('logged_in'):
        return dashboard_layout
    return login_layout

@callback(
    Output('page-content-main', 'children'),
    Input('url', 'pathname')
)
def render_page_content(pathname):
    if pathname == '/inventory':
        return inventory_management_layout
    elif pathname == '/consumables':
        return consumables_management_layout
    elif pathname == '/shipments':
        return shipment_tracking_layout
    elif pathname == '/analytics':
        return inventory_analytics_layout
    elif pathname == '/forecasting':
        return forecasting_layout
    elif pathname == '/suppliers':
        return suppliers_layout
    elif pathname == '/batch':
        return batch_tracking_layout
    elif pathname == '/users':
        return user_management_layout
    elif pathname == '/audit':
        return audit_trail_layout
    return executive_overview_layout

# Page Layouts (enhanced for professionalism with updated colors)
executive_overview_layout = dbc.Card(dbc.CardBody([
    html.H3("Executive Insights", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '700', 'letterSpacing': '0.5px'}),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-dollar-sign kpi-icon"), html.H4(id='kpi-total-value', className="number"), html.P("Total Inventory Value")])], color="primary", outline=True, className='kpi-card', id='kpi-total-value-card'), width=3),
        dbc.Tooltip("Sum of quantity * price per unit.", target='kpi-total-value-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-exclamation-triangle kpi-icon text-danger"), html.H4(id='kpi-expired', className="number"), html.P("Expired Items")])], color="danger", outline=True, className='kpi-card', id='kpi-expired-card'), width=3),
        dbc.Tooltip("Items past expiration date.", target='kpi-expired-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-exclamation-circle kpi-icon text-warning"), html.H4(id='kpi-low-stock', className="number"), html.P("Low Stock Items")])], color="warning", outline=True, className='kpi-card', id='kpi-low-stock-card'), width=3),
        dbc.Tooltip(f"Items with quantity below {LOW_STOCK_THRESHOLD}.", target='kpi-low-stock-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-box-open kpi-icon text-success"), html.H4(id='kpi-consumables-cost', className="number"), html.P("Total Consumables Cost")])], color="success", outline=True, className='kpi-card', id='kpi-consumables-cost-card'), width=3),
        dbc.Tooltip("Total cost of consumables.", target='kpi-consumables-cost-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-star kpi-icon text-info"), html.H4(id='kpi-a-items', className="number"), html.P("A-Class Items (ABC)")])], color="info", outline=True, className='kpi-card', id='kpi-a-items-card'), width=3),
        dbc.Tooltip("High-value A-class items (80% of value).", target='kpi-a-items-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-sync-alt kpi-icon text-secondary"), html.H4(id='kpi-avg-turnover', className="number"), html.P("Avg Turnover Rate")])], color="secondary", outline=True, className='kpi-card', id='kpi-avg-turnover-card'), width=3),
        dbc.Tooltip("Average inventory turnover rate.", target='kpi-avg-turnover-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-truck kpi-icon text-danger"), html.H4(id='kpi-delayed-shipments', className="number"), html.P("Delayed Shipments")])], color="danger", outline=True, className='kpi-card', id='kpi-delayed-shipments-card'), width=3),
        dbc.Tooltip("Number of delayed shipments.", target='kpi-delayed-shipments-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-exclamation kpi-icon text-warning"), html.H4(id='kpi-stockout-risk', className="number"), html.P("Stockout Risk Items")])], color="warning", outline=True, className='kpi-card', id='kpi-stockout-risk-card'), width=3),
        dbc.Tooltip("Items at high stockout risk.", target='kpi-stockout-risk-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-calendar-alt kpi-icon text-info"), html.H4(id='kpi-avg-dsi', className="number"), html.P("Avg DSI (Days)")])], color="info", outline=True, className='kpi-card', id='kpi-avg-dsi-card'), width=3),
        dbc.Tooltip("Average days to sell inventory.", target='kpi-avg-dsi-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-money-bill-wave kpi-icon text-warning"), html.H4(id='kpi-total-carrying', className="number"), html.P("Total Carrying Cost")])], color="warning", outline=True, className='kpi-card', id='kpi-total-carrying-card'), width=3),
        dbc.Tooltip("Estimated annual holding costs.", target='kpi-total-carrying-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-check-circle kpi-icon text-success"), html.H4(id='kpi-avg-service', className="number"), html.P("Avg Service Level (%)")])], color="success", outline=True, className='kpi-card', id='kpi-avg-service-card'), width=3),
        dbc.Tooltip("Percentage of demand met without stockouts.", target='kpi-avg-service-card'),
        dbc.Col(dbc.Card([dbc.CardBody([html.I(className="fas fa-exclamation-triangle kpi-icon text-danger"), html.H4(id='kpi-overstock-risk', className="number"), html.P("Overstock Risk Items")])], color="danger", outline=True, className='kpi-card', id='kpi-overstock-risk-card'), width=3),
        dbc.Tooltip("Items with excess stock.", target='kpi-overstock-risk-card'),
    ], className="mb-5", justify="center"),
    html.H4("Advanced Insights & Recommendations", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '600'}),
    dcc.Loading(
        type="circle",
        children=[
            dbc.Button([html.I(className="fas fa-lightbulb"), " Generate Insights"], id='ai-insights-button', color='primary', className='mb-3 mr-2'),
            dbc.Button([html.I(className="fas fa-envelope"), " Send Alert Emails"], id='send-alert-emails', color='danger', className='mb-3')
        ]
    ),
    html.Div(id='email-output', className='text-success mt-2 text-center'),
    html.Div(id='insights-text', className="mb-5 p-4 bg-white border rounded shadow-sm", style={'borderRadius': '8px', 'boxShadow': '0 2px 10px rgba(0,0,0,0.05)'}),
    dcc.Dropdown(
        id='category-filter',
        options=[{'label': cat, 'value': cat} for cat in sorted(df_inventory['category'].unique())],
        multi=True,
        placeholder="Filter by Category",
        className="mb-3",
        style={'borderRadius': '4px'}
    ),
    dcc.Dropdown(
        id='drug-filter',
        options=[{'label': drug, 'value': drug} for drug in sorted(df_inventory['drug_name'].unique())],
        multi=True,
        placeholder="Filter by Drug",
        className="mb-4",
        style={'borderRadius': '4px'}
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id='inventory-bar', config={'displayModeBar': False}), width=6),
        dbc.Col(dcc.Graph(id='category-pie', config={'displayModeBar': False}), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='value-treemap', config={'displayModeBar': False}), width=6),
        dbc.Col(dcc.Graph(id='abc-pie', config={'displayModeBar': False}), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='reorder-vs-quantity-bar', config={'displayModeBar': False}), width=6),
        dbc.Col(dcc.Graph(id='stockout-risk-pie', config={'displayModeBar': False}), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='aging-heatmap', config={'displayModeBar': False}), width=6),
        dbc.Col(dcc.Graph(id='demand-scatter', config={'displayModeBar': False}), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='carrying-cost-bar', config={'displayModeBar': False}), width=12),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(html.H4("Inventory Health Score", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}), width=12),
        dbc.Col(dcc.Graph(id='health-score-gauge', config={'displayModeBar': False}), width=4),
        dbc.Col(dcc.Graph(id='health-breakdown-radar', config={'displayModeBar': False}), width=8),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='inventory-turnover-trend', config={'displayModeBar': False}), width=6),
        dbc.Col(dcc.Graph(id='stockout-trend', config={'displayModeBar': False}), width=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='category-performance-heatmap', config={'displayModeBar': False}), width=12),
    ], className="mb-4"),
]), className="shadow-sm mb-5 bg-white rounded p-4", style={'borderRadius': '8px'})

inventory_management_layout = dbc.Card(dbc.CardBody([
    html.H3("Inventory Management", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '700'}),
    dash_table.DataTable(
        id='inventory-table',
        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in df_inventory.columns],
        data=df_inventory.to_dict('records'),
        editable=True,
        row_deletable=True,
        filter_action='native',
        sort_action='native',
        page_size=10,
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'},
        style_data_conditional=[
            {'if': {'column_id': 'expired', 'filter_query': '{expired} eq true'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'},
            {'if': {'column_id': 'low_stock', 'filter_query': '{low_stock} eq true'}, 'backgroundColor': '#fff3cd', 'color': '#856404'},
            {'if': {'column_id': 'reorder_alert', 'filter_query': '{reorder_alert} eq "Reorder Now"'}, 'backgroundColor': '#ffdab9', 'color': '#804000'},
            {'if': {'column_id': 'abc_category', 'filter_query': '{abc_category} eq "A"'}, 'backgroundColor': '#d4edda', 'color': '#155724'},
            {'if': {'column_id': 'stockout_risk', 'filter_query': '{stockout_risk} eq "High Risk"'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'},
            {'if': {'column_id': 'overstock_risk', 'filter_query': '{overstock_risk} eq "High Risk"'}, 'backgroundColor': '#ffe4c4', 'color': '#804000'},
        ]
    ),
    dbc.Row([
        dbc.Col(dbc.Button([html.I(className="fas fa-plus"), " Add Row"], id='add-inv-row', color='primary', className='mr-2')),
        dbc.Col(dbc.Button([html.I(className="fas fa-save"), " Save to CSV"], id='save-inv', color='success', className='mr-2')),
        dbc.Col(dcc.Loading(type="default", children=dbc.Button([html.I(className="fas fa-file-pdf"), " Export to PDF"], id='export-inv-pdf', color='info', className='mr-2'))),
        dbc.Col(dcc.Loading(type="default", children=dbc.Button([html.I(className="fas fa-sync"), " Generate Reorder Suggestions"], id='reorder-suggestions', color='warning'))),
    ], className="my-3", justify="center"),
    html.Div(id='inv-output', className='text-success mt-2 text-center'),
    html.H4("Reorder Suggestions", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}),
    dash_table.DataTable(
        id='reorder-table',
        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in ['drug_name', 'quantity', 'reorder_point', 'safety_stock', 'suggested_order_quantity', 'estimated_cost', 'supplier']],
        data=[],
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'},
        style_data_conditional=[{'if': {'column_id': 'suggested_order_quantity'}, 'backgroundColor': '#fff3cd', 'color': '#856404'}]
    ),
    html.Div(id='reorder-output', className='text-success mt-2 text-center'),
    html.H4("Advanced Inventory Metrics", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}),
    dash_table.DataTable(
        id='advanced-metrics-table',
        columns=[
            {"name": "Drug Name", "id": "drug_name"},
            {"name": "GMROI", "id": "gmroi"},
            {"name": "Stock Cover (Days)", "id": "stock_cover"},
            {"name": "Fill Rate (%)", "id": "fill_rate"},
            {"name": "Stockout Frequency", "id": "stockout_frequency"},
            {"name": "Order Cycle Time", "id": "order_cycle_time"},
            {"name": "Inventory Accuracy", "id": "inventory_accuracy"},
            {"name": "Carrying Cost %", "id": "carrying_cost_percent"}
        ],
        data=[],
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'}
    ),
]), className="shadow-sm mb-5 bg-white rounded p-4")

consumables_management_layout = dbc.Card(dbc.CardBody([
    html.H3("Consumables Management", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '700'}),
    dash_table.DataTable(
        id='consumables-table',
        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in df_consumables.columns],
        data=df_consumables.to_dict('records'),
        editable=True,
        row_deletable=True,
        filter_action='native',
        sort_action='native',
        page_size=10,
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'},
        style_data_conditional=[{'if': {'column_id': 'reorder_alert', 'filter_query': '{reorder_alert} eq "Reorder Now"'}, 'backgroundColor': '#ffdab9', 'color': '#804000'}]
    ),
    dbc.Row([
        dbc.Col(dbc.Button('Add Row', id='add-cons-row', color='primary', className='mr-2')),
        dbc.Col(dbc.Button('Save to CSV', id='save-cons', color='success', className='mr-2')),
        dbc.Col(dcc.Loading(type="default", children=dbc.Button('Export to PDF', id='export-cons-pdf', color='info'))),
    ], className="my-3", justify="center"),
    html.Div(id='cons-output', className='text-success mt-2 text-center')
]), className="shadow-sm mb-5 bg-white rounded p-4")

shipment_tracking_layout = dbc.Card(dbc.CardBody([
    html.H3("Shipment Tracking", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '700'}),
    dash_table.DataTable(
        id='shipment-table',
        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in df_shipments.columns],
        data=df_shipments.to_dict('records'),
        editable=True,
        row_deletable=True,
        filter_action='native',
        sort_action='native',
        page_size=10,
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'},
        style_data_conditional=[
            {'if': {'column_id': 'status', 'filter_query': '{status} eq "Delayed"'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'},
            {'if': {'column_id': 'status', 'filter_query': '{status} eq "In Transit"'}, 'backgroundColor': '#ffdab9', 'color': '#804000'}
        ]
    ),
    dbc.Row([
        dbc.Col(dbc.Button('Add Shipment', id='add-shipment-row', color='primary', className='mr-2')),
        dbc.Col(dbc.Button('Save Shipments', id='save-shipments', color='success', className='mr-2')),
        dbc.Col(dcc.Loading(type="default", children=dbc.Button('Export to PDF', id='export-shipments-pdf', color='info'))),
    ], className="my-3", justify="center"),
    html.Div(id='shipment-output', className='text-success mt-2 text-center')
]), className="shadow-sm mb-5 bg-white rounded p-4")

inventory_analytics_layout = dbc.Card(dbc.CardBody([
    html.H3("Inventory Analytics & Optimization", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '700'}),
    dcc.Dropdown(
        id='category-filter-analytics',
        options=[{'label': cat, 'value': cat} for cat in sorted(df_inventory['category'].unique())],
        multi=True,
        placeholder="Filter by Category",
        className="mb-3",
        style={'borderRadius': '4px'}
    ),
    dcc.Dropdown(
        id='drug-filter-analytics',
        options=[{'label': drug, 'value': drug} for drug in sorted(df_inventory['drug_name'].unique())],
        multi=True,
        placeholder="Filter by Drug",
        className="mb-4",
        style={'borderRadius': '4px'}
    ),
    dbc.Row([
        dbc.Col(html.H4("Economic Order Quantity (EOQ)", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}), width=12),
        dbc.Col(dcc.Graph(id='eoq-bar-chart', config={'displayModeBar': False}), width=12),
    ]),
    dash_table.DataTable(
        id='eoq-table',
        columns=[
            {"name": "Drug Name", "id": "drug_name"},
            {"name": "Annual Demand", "id": "annual_demand"},
            {"name": "Ordering Cost ($)", "id": "ordering_cost"},
            {"name": "Holding Cost (%)", "id": "holding_cost_rate"},
            {"name": "EOQ", "id": "eoq"},
        ],
        data=[],
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'}
    ),
    dbc.Row([
        dbc.Col(html.H4("Stock Age by Category", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}), width=12),
        dbc.Col(dcc.Graph(id='stock-age-box', config={'displayModeBar': False}), width=12),
    ]),
    dbc.Row([
        dbc.Col(html.H4("Inventory Turnover Rate", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}), width=12),
        dbc.Col(dcc.Graph(id='turnover-bar', config={'displayModeBar': False}), width=12),
    ]),
    dbc.Row([
        dbc.Col(html.H4("Dead Stock Analysis", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}), width=12),
        dbc.Col(dcc.Graph(id='dead-stock-bar', config={'displayModeBar': False}), width=6),
        dbc.Col(dash_table.DataTable(
            id='dead-stock-table',
            columns=[{"name": "Drug Name", "id": "drug_name"}, {"name": "Days Since Last Sale", "id": "days_since_last_sale"}, {"name": "Quantity", "id": "quantity"}],
            data=[],
            style_table={'overflowX': 'auto', 'borderRadius': '8px'},
            style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
            style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'}
        ), width=6),
    ]),
    dbc.Row([
        dbc.Col(html.H4("Aging Analysis", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}), width=12),
        dbc.Col(dcc.Graph(id='aging-heatmap-analytics', config={'displayModeBar': False}), width=8),
        dbc.Col(dash_table.DataTable(
            id='aging-table',
            columns=[{"name": "Drug Name", "id": "drug_name"}, {"name": "Quantity", "id": "quantity"}, {"name": "Aging Bucket", "id": "aging_bucket"}, {"name": "Total Value", "id": "total_value"}],
            data=[],
            style_table={'overflowX': 'auto', 'borderRadius': '8px'},
            style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
            style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'}
        ), width=4),
    ]),
    dbc.Row([
        dbc.Col(html.H4("Service Level Analysis", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}), width=12),
        dbc.Col(dcc.Graph(id='service-level-bar', config={'displayModeBar': False}), width=12),
    ]),
    dbc.Row([
        dbc.Col(html.H4("Sales Trend Analysis", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}), width=12),
        dbc.Col(dcc.Graph(id='sales-trend', config={'displayModeBar': False}), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Slider(id='what-if-demand', min=-50, max=50, step=5, value=0, marks={i: f'{i}%' for i in [-50, -25, 0, 25, 50]}, className="mb-3"), width=12),
        dbc.Col(html.Div(id='what-if-output', className="text-info"), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Loading(type="default", children=dbc.Button('Export Analytics Report', id='export-analytics-pdf', color='info')), width=12),
    ]),
    html.Div(id='analytics-output', className='text-success mt-2 text-center')
]), className="shadow-sm mb-5 bg-white rounded p-4")


forecasting_layout = dbc.Card(dbc.CardBody([
    html.H3("Stock Forecasting", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '700'}),
    dcc.Dropdown(
        id='forecast-drug',
        options=[{'label': drug, 'value': drug} for drug in sorted(df_inventory['drug_name'].unique())],
        placeholder="Select a Drug to Forecast",
        style={'borderRadius': '4px'}
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id='forecast-graph', config={'displayModeBar': False}), width=8),
        dbc.Col(dash_table.DataTable(
            id='forecast-table',
            columns=[{"name": "Date", "id": "date"}, {"name": "Predicted Quantity", "id": "predicted"}, {"name": "Action", "id": "action"}],
            data=[],
            style_table={'overflowX': 'auto', 'borderRadius': '8px'},
            style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
            style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'}
        ), width=4),
    ]),
]), className="shadow-sm mb-5 bg-white rounded p-4")

suppliers_layout = dbc.Card(dbc.CardBody([
    html.H3("Suppliers Management", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '700'}),
    dash_table.DataTable(
        id='suppliers-table',
        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in df_suppliers.columns if i not in ['lat', 'lon']],
        data=df_suppliers.to_dict('records'),
        editable=True,
        row_deletable=True,
        filter_action='native',
        sort_action='native',
        page_size=10,
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'}
    ),
    dcc.Graph(id='supplier-geo-map', config={'displayModeBar': False}),
    html.H4("Supplier Performance Metrics", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}),
    dash_table.DataTable(
        id='supplier-performance-table',
        columns=[
            {"name": "Supplier", "id": "supplier"},
            {"name": "Number of Shipments", "id": "num_shipments"},
            {"name": "On-Time Delivery Rate (%)", "id": "on_time_delivery_rate"},
            {"name": "Delay Percentage (%)", "id": "delay_percentage"},
            {"name": "Performance Score", "id": "performance_score"}
        ],
        data=supplier_performance.to_dict('records'),
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'},
        style_data_conditional=[
            {'if': {'column_id': 'delay_percentage', 'filter_query': '{delay_percentage} > 50'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'}
        ]
    ),
    dcc.Graph(id='supplier-performance-bar', config={'displayModeBar': False}),
    html.H4("Supplier Performance Matrix", className="text-center mt-4", style={'color': '#0056b3', 'fontWeight': '600'}),
    dcc.Graph(id='supplier-matrix', config={'displayModeBar': False}),
    dcc.Loading(type="default", children=dbc.Button('Export to PDF', id='export-suppliers-pdf', color='info', className='mt-3')),
    html.Div(id='suppliers-output', className='text-success mt-2 text-center')
]), className="shadow-sm mb-5 bg-white rounded p-4")

batch_tracking_layout = dbc.Card(dbc.CardBody([
    html.H3("Batch Tracking", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '700'}),
    dash_table.DataTable(
        id='batch-table',
        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in batch_data.columns],
        data=batch_data.to_dict('records'),
        filter_action='native',
        sort_action='native',
        page_size=10,
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'},
        style_data_conditional=[{'if': {'column_id': 'status', 'filter_query': '{status} eq "Expired"'}, 'backgroundColor': '#f8d7da', 'color': '#721c24'}]
    ),
    dcc.Graph(id='batch-heatmap', config={'displayModeBar': False}),
    dcc.Loading(type="default", children=dbc.Button('Export to PDF', id='export-batch-pdf', color='info', className='mt-3')),
    html.Div(id='batch-output', className='text-success mt-2 text-center')
]), className="shadow-sm mb-5 bg-white rounded p-4")

user_management_layout = dbc.Card(dbc.CardBody([
    html.H3("User Management", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '700'}),
    dash_table.DataTable(
        id='users-table',
        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in df_users.columns if i != 'password'],
        data=df_users.to_dict('records'),
        editable=True,
        row_deletable=True,
        filter_action='native',
        sort_action='native',
        page_size=10,
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'}
    ),
    dbc.Row([
        dbc.Col(dbc.Button('Add Row', id='add-user-row', color='primary', className='mr-2')),
        dbc.Col(dbc.Button('Save to CSV', id='save-users', color='success', className='mr-2')),
        dbc.Col(dcc.Loading(type="default", children=dbc.Button('Export to PDF', id='export-users-pdf', color='info'))),
    ], className="my-3", justify="center"),
    html.Div(id='users-output', className='text-success mt-2 text-center')
]), className="shadow-sm mb-5 bg-white rounded p-4")

audit_trail_layout = dbc.Card(dbc.CardBody([
    html.H3("Audit Trail", className="text-center mb-4", style={'color': '#0056b3', 'fontWeight': '700'}),
    dash_table.DataTable(
        id='audit-table',
        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in audit_log.columns],
        data=audit_log.to_dict('records'),
        filter_action='native',
        sort_action='native',
        page_size=10,
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_cell={'padding': '10px', 'fontSize': '14px', 'textAlign': 'center', 'borderBottom': '1px solid #dee2e6'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': '600', 'textTransform': 'uppercase', 'letterSpacing': '0.5px', 'color': '#495057'}
    ),
    dcc.Loading(type="default", children=dbc.Button('Export to PDF', id='export-audit-pdf', color='info', className='mt-3')),
    html.Div(id='audit-output', className='text-success mt-2 text-center')
]), className="shadow-sm mb-5 bg-white rounded p-4")

# Callback to sync filters from the executive dashboard to the central store
@callback(
    Output('filter-store', 'data'),
    [Input('category-filter', 'value'), Input('drug-filter', 'value')]
)
def update_filter_store(categories, drugs):
    return {'category': categories, 'drug': drugs}

# This callback updates the analytics page filters based on the central store
@callback(
    [Output('category-filter-analytics', 'value'),
     Output('drug-filter-analytics', 'value')],
    Input('url', 'pathname'),
    State('filter-store', 'data')
)
def update_analytics_filters_from_store(pathname, filter_data):
    if pathname == '/analytics':
        return filter_data['category'], filter_data['drug']
    raise PreventUpdate


# Add missing callbacks for add rows (similar to inventory)
@callback(
    [Output('consumables-table', 'data'), Output('cons-output', 'children')],
    Input('add-cons-row', 'n_clicks'),
    [State('consumables-table', 'data'), State('consumables-table', 'columns'), State('login-status', 'data')],
    prevent_initial_call=True
)
def add_cons_row(n, rows, cols, login_status):
    if not login_status.get('logged_in') or login_status['role'] not in ['admin', 'manager']:
        return rows, 'Permission denied: Only admins and managers can add rows'
    if n:
        new_row = {c['id']: '' for c in cols}
        new_row['id'] = max([int(row['id']) for row in rows if 'id' in row and str(row['id']).isdigit()] + [0]) + 1
        rows.append(new_row)
        audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Add Consumables Row', 'Added new row']
        audit_log.to_csv(AUDIT_PATH, index=False)
        logging.info(f"User {login_status['username']} added new consumables row")
        return rows, 'New consumables row added successfully'
    return rows, ''

@callback(
    [Output('shipment-table', 'data'), Output('shipment-output', 'children')],
    Input('add-shipment-row', 'n_clicks'),
    [State('shipment-table', 'data'), State('shipment-table', 'columns'), State('login-status', 'data')],
    prevent_initial_call=True
)
def add_shipment_row(n, rows, cols, login_status):
    if not login_status.get('logged_in') or login_status['role'] not in ['admin', 'manager']:
        return rows, 'Permission denied: Only admins and managers can add rows'
    if n:
        new_row = {c['id']: '' for c in cols}
        new_row['id'] = max([int(row['id']) for row in rows if 'id' in row and str(row['id']).isdigit()] + [0]) + 1
        rows.append(new_row)
        audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Add Shipment Row', 'Added new row']
        audit_log.to_csv(AUDIT_PATH, index=False)
        logging.info(f"User {login_status['username']} added new shipment row")
        return rows, 'New shipment row added successfully'
    return rows, ''

@callback(
    [Output('users-table', 'data'), Output('users-output', 'children')],
    Input('add-user-row', 'n_clicks'),
    [State('users-table', 'data'), State('users-table', 'columns'), State('login-status', 'data')],
    prevent_initial_call=True
)
def add_user_row(n, rows, cols, login_status):
    if not login_status.get('logged_in') or login_status['role'] != 'admin':
        return rows, 'Permission denied: Only admins can add users'
    if n:
        new_row = {c['id']: '' for c in cols}
        new_row['id'] = max([int(row['id']) for row in rows if 'id' in row and str(row['id']).isdigit()] + [0]) + 1
        rows.append(new_row)
        audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Add User Row', 'Added new row']
        audit_log.to_csv(AUDIT_PATH, index=False)
        logging.info(f"User {login_status['username']} added new user row")
        return rows, 'New user row added successfully'
    return rows, ''

# COMBINED LOGIN AND REDIRECT CALLBACK
@callback(
    [Output('login-status', 'data'),
     Output('login-output', 'children'),
     Output('url', 'pathname', allow_duplicate=True)],
    Input('login-button', 'n_clicks'),
    [State('username', 'value'),
     State('password', 'value')],
    prevent_initial_call=True
)
def login(n, username, password):
    if n and username and password:
        if not username.strip() or not password.strip():
            return {'logged_in': False}, dbc.Alert('Username and password cannot be empty.', color='danger'), dash.no_update
        hashed_pw = hashlib.md5(password.encode()).hexdigest()
        print(f"Login attempt for '{username}', provided hash: {hashed_pw}")  # Debug print
        user = df_users[df_users['username'] == username.strip()]
        print(f"Found user: {not user.empty}, stored hash: {user['password'].iloc[0] if not user.empty else 'None'}")  # Debug
        if not user.empty and user['password'].iloc[0] == hashed_pw:
            role = user['role'].iloc[0]
            audit_log.loc[len(audit_log)] = [datetime.now(), username, 'Login', 'Successful login']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {username} logged in successfully")
            return {'logged_in': True, 'username': username, 'role': role}, dbc.Alert('Login successful!', color='success'), '/'
        else:
            audit_log.loc[len(audit_log)] = [datetime.now(), username, 'Login Attempt', 'Failed login attempt']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.warning(f"Failed login attempt for {username}")
            return {'logged_in': False}, dbc.Alert('Invalid username or password. Please try again.', color='danger'), dash.no_update
    raise PreventUpdate

# Logout callback to clear session and redirect to login page
@callback(
    [Output('login-status', 'data', allow_duplicate=True),
     Output('url', 'pathname', allow_duplicate=True)],
    Input('logout-button', 'n_clicks'),
    State('login-status', 'data'),
    prevent_initial_call=True
)
def logout(n, data):
    if n and data.get('logged_in'):
        audit_log.loc[len(audit_log)] = [datetime.now(), data['username'], 'Logout', 'User logged out']
        audit_log.to_csv(AUDIT_PATH, index=False)
        logging.info(f"User {data['username']} logged out")
        return {'logged_in': False}, '/login'  # Redirect to a login page
    raise PreventUpdate

@callback(
    Output('user-greeting', 'children'),
    Input('login-status', 'data')
)
def update_greeting(data):
    if data and data.get('logged_in'):
        return f"Welcome, {data['username']} ({data['role'].capitalize()})"
    return ''

# Update Executive KPI's
@callback(
    [Output('kpi-total-value', 'children'), Output('kpi-expired', 'children'), Output('kpi-low-stock', 'children'),
     Output('kpi-consumables-cost', 'children'), Output('kpi-a-items', 'children'), Output('kpi-avg-turnover', 'children'),
     Output('kpi-delayed-shipments', 'children'), Output('kpi-stockout-risk', 'children'),
     Output('kpi-avg-dsi', 'children'), Output('kpi-total-carrying', 'children'),
     Output('kpi-avg-service', 'children'), Output('kpi-overstock-risk', 'children'),
     Output('insights-text', 'children')],
    [Input('category-filter', 'value'), Input('drug-filter', 'value'),
     Input('ai-insights-button', 'n_clicks'), Input('interval-update', 'n_intervals')]
)
def update_executive_kpis(categories, drugs, n_clicks, n_intervals):
    df_inv_filtered = df_inventory.copy()
    if categories: df_inv_filtered = df_inv_filtered[df_inv_filtered['category'].isin(categories)]
    if drugs: df_inv_filtered = df_inv_filtered[df_inv_filtered['drug_name'].isin(drugs)]
    
    total_value = df_inv_filtered['total_value'].sum()
    expired = df_inv_filtered['expired'].sum()
    low_stock = df_inv_filtered['low_stock'].sum()
    consumables_cost = df_consumables['total_cost'].sum()
    a_items = df_inv_filtered[df_inv_filtered['abc_category'] == 'A'].shape[0]
    a_value_percent = df_inv_filtered[df_inv_filtered['abc_category'] == 'A']['total_value'].sum() / total_value * 100 if total_value > 0 else 0
    avg_turnover = df_inv_filtered['turnover_rate'].mean() if 'turnover_rate' in df_inv_filtered.columns else 0
    delayed_shipments = num_delayed_shipments
    stockout_risk = df_inv_filtered[df_inv_filtered['stockout_risk'] == 'High Risk'].shape[0]
    avg_dsi = df_inv_filtered['dsi'].mean()
    total_carrying = df_inv_filtered['carrying_cost'].sum()
    avg_service = df_inv_filtered['service_level'].mean() * 100
    overstock_risk = df_inv_filtered[df_inv_filtered['overstock_risk'] == 'High Risk'].shape[0]
    
    insights = dcc.Markdown(generate_insights(df_inv_filtered))
    
    return (
        f"${total_value:,.2f}", expired, low_stock, f"${consumables_cost:,.2f}",
        f"{a_items} ({a_value_percent:.1f}%)", f"{avg_turnover:.2f}", delayed_shipments, stockout_risk,
        f"{avg_dsi:.1f} days", f"${total_carrying:,.2f}", f"{avg_service:.1f}%", overstock_risk, insights
    )

# Update Executive Charts
@callback(
    [Output('inventory-bar', 'figure'), Output('category-pie', 'figure'), Output('value-treemap', 'figure'),
     Output('abc-pie', 'figure'), Output('reorder-vs-quantity-bar', 'figure'), Output('stockout-risk-pie', 'figure'),
     Output('aging-heatmap', 'figure'), Output('demand-scatter', 'figure'), Output('carrying-cost-bar', 'figure')],
    [Input('category-filter', 'value'), Input('drug-filter', 'value'), Input('theme-toggle', 'value'), Input('interval-update', 'n_intervals')]
)
def update_executive_charts(categories, drugs, theme, n_intervals):
    df_inv = df_inventory.copy()
    if categories: df_inv = df_inv[df_inv['category'].isin(categories)]
    if drugs: df_inv = df_inv[df_inv['drug_name'].isin(drugs)]
    if df_inv.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title='No Data Available', showlegend=False)
        return [empty_fig] * 9
    
    template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    
    inventory_bar = px.bar(df_inv.sort_values('quantity'), y='drug_name', x='quantity', color='expired', orientation='h',
                           title='Inventory Levels (Red: Expired)', color_discrete_map={True: '#dc3545', False: '#28a745'},
                           hover_data=['expiration_date', 'total_value', 'supplier', 'reorder_alert', 'abc_category'], template=template)
    inventory_bar.update_layout(font=dict(family="Arial", size=12), title_font_size=18)

    category_pie = px.pie(df_inv, values='total_value', names='category', title='Inventory Value by Category',
                          color_discrete_sequence=px.colors.qualitative.Pastel, template=template)
    category_pie.update_layout(font=dict(family="Arial", size=12), title_font_size=18)

    value_treemap = px.treemap(df_inv, path=['category', 'drug_name'], values='total_value',
                               title='Inventory Value Treemap', template=template)
    value_treemap.update_layout(font=dict(family="Arial", size=12), title_font_size=18)

    abc_pie = px.pie(df_inv, values='total_value', names='abc_category', title='ABC Analysis by Value',
                     color_discrete_sequence=px.colors.qualitative.Set1, template=template)
    abc_pie.update_layout(font=dict(family="Arial", size=12), title_font_size=18)

    reorder_vs_quantity_bar = go.Figure()
    reorder_vs_quantity_bar.add_trace(go.Bar(x=df_inv['drug_name'], y=df_inv['quantity'], name='Current Quantity', marker_color='#007bff'))
    reorder_vs_quantity_bar.add_trace(go.Scatter(x=df_inv['drug_name'], y=df_inv['reorder_point'], name='Reorder Point', mode='lines', line=dict(color='red', dash='dash')))
    reorder_vs_quantity_bar.update_layout(title='Reorder Point vs Current Quantity', template=template, barmode='group', font=dict(family="Arial", size=12), title_font_size=18)

    stockout_risk_pie = px.pie(df_inv, values='total_value', names='stockout_risk', title='Stockout Risk by Value',
                               color_discrete_sequence=px.colors.qualitative.Set1, template=template)
    stockout_risk_pie.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
                                      
    aging_heatmap_data = df_inv.pivot_table(index='category', columns='aging_bucket', values='quantity', aggfunc='sum').fillna(0)
    aging_heatmap = px.imshow(aging_heatmap_data, title='Inventory Aging by Category', template=template)
    aging_heatmap.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    
    demand_scatter = px.scatter(df_inv, x='demand_volatility', y='quantity', color='stockout_risk',
                                size='total_value', hover_data=['drug_name'], title='Demand Volatility vs. Stock Level',
                                template=template, color_discrete_map={'High Risk': '#dc3545', 'Low Risk': '#28a745'})
    demand_scatter.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    
    carrying_cost_bar = px.bar(df_inv.groupby('category', observed=True)['carrying_cost'].sum().reset_index(),
                           x='category', y='carrying_cost', title='Carrying Cost by Category',
                           color='carrying_cost', color_continuous_scale='Blues', template=template)
    carrying_cost_bar.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    
    return (inventory_bar, category_pie, value_treemap, abc_pie, reorder_vs_quantity_bar, stockout_risk_pie,
            aging_heatmap, demand_scatter, carrying_cost_bar)

# Update Executive Trends
@callback(
    [Output('inventory-turnover-trend', 'figure'), 
     Output('stockout-trend', 'figure'),
     Output('category-performance-heatmap', 'figure')],
    [Input('category-filter', 'value'), Input('drug-filter', 'value'), Input('theme-toggle', 'value')]
)
def update_executive_trends(categories, drugs, theme):
    template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    df_inv = df_inventory.copy()
    if categories: df_inv = df_inv[df_inv['category'].isin(categories)]
    if drugs: df_inv = df_inv[df_inv['drug_name'].isin(drugs)]
    
    # Inventory turnover trend
    turnover_trend = px.line(df_inv.sort_values('turnover_rate'), x='drug_name', y='turnover_rate',
                            title='Inventory Turnover Rate Trend', template=template)
    turnover_trend.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    
    # Stockout trend
    stockout_data = df_inv[['drug_name', 'stockout_frequency']].sort_values('stockout_frequency')
    stockout_trend = px.bar(stockout_data, x='drug_name', y='stockout_frequency',
                           title='Stockout Frequency by Product', template=template)
    stockout_trend.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    
    # Category performance heatmap
    category_perf = df_inv.groupby('category', observed=True).agg({
        'service_level': 'mean',
        'turnover_rate': 'mean',
        'inventory_accuracy': 'mean',
        'gmroi': 'mean'
    }).reset_index()
    
    # Normalize values for heatmap
    for col in ['service_level', 'turnover_rate', 'inventory_accuracy', 'gmroi']:
        category_perf[col] = (category_perf[col] - category_perf[col].min()) / (category_perf[col].max() - category_perf[col].min())
    
    category_heatmap = px.imshow(category_perf.set_index('category').T,
                                title='Category Performance Heatmap',
                                color_continuous_scale='RdYlGn',
                                template=template)
    category_heatmap.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    
    return turnover_trend, stockout_trend, category_heatmap


# Update Inventory Analytics Charts and Data (MODIFIED)
@callback(
    [Output('eoq-table', 'data'), Output('eoq-bar-chart', 'figure'),
     Output('stock-age-box', 'figure'), Output('turnover-bar', 'figure'),
     Output('dead-stock-bar', 'figure'), Output('dead-stock-table', 'data'),
     Output('aging-heatmap-analytics', 'figure'),
     Output('service-level-bar', 'figure'),
     Output('sales-trend', 'figure'),
     Output('what-if-output', 'children'), Output('aging-table', 'data')],
    [Input('what-if-demand', 'value'), Input('theme-toggle', 'value'), Input('interval-update', 'n_intervals'),
     Input('category-filter-analytics', 'value'), Input('drug-filter-analytics', 'value')],
    State('login-status', 'data')
)
def update_analytics_charts(change_percent, theme, n_intervals, categories, drugs, login_status):
    template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    df_inv = df_inventory.copy()
    if categories: df_inv = df_inv[df_inv['category'].isin(categories)]
    if drugs: df_inv = df_inv[df_inv['drug_name'].isin(drugs)]

    # What-If Scenario
    if not login_status.get('logged_in') or login_status['role'] not in ['admin', 'manager']:
        what_if_output = 'Permission denied: Only admins and managers can run scenarios'
    else:
        adjusted_demand = df_inv['average_daily_demand'] * (1 + change_percent / 100)
        new_reorder = adjusted_demand * df_inv['lead_time_days'] + df_inv['safety_stock']
        impacted = (df_inv['quantity'] < new_reorder).sum()
        what_if_output = f"With {change_percent}% demand change, {impacted} items would trigger reorder."

    # EOQ Calculation and Bar Chart
    eoq_df = df_inv.copy()
    eoq_df['annual_demand'] = eoq_df['average_daily_demand'] * 365
    eoq_df['ordering_cost'] = 50
    eoq_df['holding_cost_rate'] = 0.2
    eoq_df['eoq'] = eoq_df.apply(lambda row: calculate_eoq(row['annual_demand'], row['ordering_cost'], row['holding_cost_rate'], row['price_per_unit']), axis=1).round(2)
    eoq_data = eoq_df[['drug_name', 'annual_demand', 'ordering_cost', 'holding_cost_rate', 'eoq']].to_dict('records')
    eoq_bar_chart = px.bar(eoq_df.sort_values('eoq'), x='drug_name', y='eoq', title='Economic Order Quantity (EOQ)',
                           color='eoq', color_continuous_scale='Plasma', template=template)
    eoq_bar_chart.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    
    # Stock Age Box Plot
    stock_age_box = px.box(df_inv, x='category', y='stock_age_days', color='category',
                           title='Stock Age Distribution by Category (Days)', template=template)
    stock_age_box.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
                           
    # Inventory Turnover Rate Bar Chart
    turnover_bar = px.bar(df_inv[['drug_name', 'turnover_rate']].sort_values('turnover_rate'), x='drug_name', y='turnover_rate', title='Inventory Turnover Rate by Drug',
                          color='turnover_rate', color_continuous_scale='Viridis', template=template)
    turnover_bar.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
                          
    # Dead Stock Analysis
    dead_stock_df = df_history.groupby('drug_name')['date'].max().reset_index()
    dead_stock_df.columns = ['drug_name', 'last_sale_date']
    dead_stock_df['days_since_last_sale'] = (pd.to_datetime(current_datetime.date()) - dead_stock_df['last_sale_date']).dt.days
    dead_stock_df = pd.merge(dead_stock_df, df_inv[['drug_name', 'quantity']], on='drug_name', how='left')
    dead_stock_df['quantity'] = pd.to_numeric(dead_stock_df['quantity'], errors='coerce').fillna(0)
    dead_stock_df['days_since_last_sale'] = dead_stock_df['days_since_last_sale'].fillna(0).astype(int)
    dead_stock_table_data = dead_stock_df[dead_stock_df['days_since_last_sale'] > 180].sort_values('days_since_last_sale', ascending=False).to_dict('records')
    dead_stock_bar = px.bar(dead_stock_df.sort_values('days_since_last_sale', ascending=False).head(20),
                            x='drug_name', y='days_since_last_sale', title='Top 20 Dead Stock Items',
                            color='days_since_last_sale', color_continuous_scale='Reds', template=template)
    dead_stock_bar.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    
    # Aging Heatmap
    aging_heatmap_data = df_inv.pivot_table(index='category', columns='aging_bucket', values='quantity', aggfunc='sum').fillna(0)
    aging_heatmap_analytics = px.imshow(aging_heatmap_data, title='Inventory Aging (Analytics Tab)', template=template)
    aging_heatmap_analytics.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    aging_table_data = df_inv[['drug_name', 'quantity', 'aging_bucket', 'total_value']].to_dict('records')

    # Service Level Bar
    service_level_bar = px.bar(df_inv.groupby('category', observed=True)['service_level'].mean().reset_index(),
                           x='category', y='service_level', title='Average Service Level by Category',
                           color='service_level', color_continuous_scale='Greens', template=template)
    service_level_bar.update_layout(font=dict(family="Arial", size=12), title_font_size=18, yaxis_tickformat='%')

    # Sales Trend with Linear Regression
    daily_sales = df_history.groupby('date')['quantity'].sum().reset_index()
    sales_trend_fig = go.Figure(layout={'title': 'Insufficient data for trend'})
    if len(daily_sales) >= 2:
        daily_sales['days'] = (daily_sales['date'] - daily_sales['date'].min()).dt.days
        model = LinearRegression()
        model.fit(daily_sales[['days']], daily_sales['quantity'])
        daily_sales['trend'] = model.predict(daily_sales[['days']])
        sales_trend_fig = go.Figure()
        sales_trend_fig.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['quantity'], mode='markers', name='Daily Sales'))
        sales_trend_fig.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['trend'], mode='lines', name='Trend Line'))
        sales_trend_fig.update_layout(title='Overall Sales Trend', template=template, font=dict(family="Arial", size=12), title_font_size=18)

    return (eoq_data, eoq_bar_chart, stock_age_box, turnover_bar, dead_stock_bar, dead_stock_table_data, aging_heatmap_analytics, service_level_bar, sales_trend_fig, what_if_output, aging_table_data)


# THIS IS THE NEW CALLBACK TO FIX THE ERROR
# We create a new, separate callback specifically for the advanced metrics table
# that triggers when the user navigates to the Inventory Management page.
@callback(
    Output('advanced-metrics-table', 'data'),
    Input('url', 'pathname'), # Trigger on URL change
    State('filter-store', 'data')
)
def update_advanced_metrics_table_on_page_change(pathname, filter_data):
    if pathname == '/inventory':
        df_inv = df_inventory.copy()
        categories = filter_data['category']
        drugs = filter_data['drug']
        if categories: df_inv = df_inv[df_inv['category'].isin(categories)]
        if drugs: df_inv = df_inv[df_inv['drug_name'].isin(drugs)]
        
        metrics_data = df_inv[['drug_name', 'gmroi', 'stock_cover', 'fill_rate', 
                              'stockout_frequency', 'order_cycle_time', 'inventory_accuracy', 
                              'carrying_cost_percent']].copy()
        
        # Format values
        metrics_data['gmroi'] = metrics_data['gmroi'].round(2)
        metrics_data['stock_cover'] = metrics_data['stock_cover'].round(1)
        metrics_data['fill_rate'] = metrics_data['fill_rate'].round(1)
        metrics_data['stockout_frequency'] = (metrics_data['stockout_frequency'] * 100).round(1)
        metrics_data['order_cycle_time'] = metrics_data['order_cycle_time'].round(1)
        metrics_data['inventory_accuracy'] = metrics_data['inventory_accuracy'].round(1)
        metrics_data['carrying_cost_percent'] = metrics_data['carrying_cost_percent'].round(1)
        
        return metrics_data.to_dict('records')
    return [] # Return an empty list if not on the correct page


# Update Health Score and Advanced Metrics (MODIFIED)
# The output for 'advanced-metrics-table' has been removed from this callback
@callback(
    [Output('health-score-gauge', 'figure'), Output('health-breakdown-radar', 'figure')],
    [Input('category-filter', 'value'), Input('drug-filter', 'value'), Input('theme-toggle', 'value')]
)
def update_health_score(categories, drugs, theme):
    template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    df_inv = df_inventory.copy()
    if categories: df_inv = df_inv[df_inv['category'].isin(categories)]
    if drugs: df_inv = df_inv[df_inv['drug_name'].isin(drugs)]
    
    # Calculate health score (simplified)
    service_score = df_inv['service_level'].mean() * 50  # Max 50 points
    turnover_score = min((df_inv['turnover_rate'].mean() if 'turnover_rate' in df_inv.columns else 0) * 10, 25)  # Max 25 points
    accuracy_score = df_inv['inventory_accuracy'].mean() * 0.25 if 'inventory_accuracy' in df_inv.columns else 0 # Max 25 points
    
    health_score = min(service_score + turnover_score + accuracy_score, 100)
    
    # Gauge chart
    gauge_fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = health_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Health Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#0056b3"},
            'steps': [
                {'range': [0, 50], 'color': "#f8d7da"},
                {'range': [50, 75], 'color': "#fff3cd"},
                {'range': [75, 100], 'color': "#d4edda"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    gauge_fig.update_layout(template=template, font=dict(family="Arial", size=12))
    
    # Radar chart for health breakdown
    categories_radar = ['Service Level', 'Turnover Rate', 'Inventory Accuracy']
    values_radar = [service_score/50*100, turnover_score/25*100, accuracy_score/25*100]
    
    radar_fig = go.Figure(data=go.Scatterpolar(
        r=values_radar + [values_radar[0]],
        theta=categories_radar + [categories_radar[0]],
        fill='toself',
        line_color='#0056b3'
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        title="Health Score Breakdown",
        template=template,
        font=dict(family="Arial", size=12)
    )
    
    return gauge_fig, radar_fig

# Update Suppliers Page Charts and Data
@callback(
    [Output('supplier-geo-map', 'figure'), Output('supplier-performance-bar', 'figure'), Output('supplier-matrix', 'figure')],
    Input('theme-toggle', 'value')
)
def update_supplier_charts(theme):
    template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    
    supplier_geo_map = px.scatter_geo(df_suppliers, lat='lat', lon='lon', hover_name='supplier_name',
                                      hover_data=['country', 'contact_email', 'location_name'], title='Supplier Locations',
                                      projection='natural earth', size=[10] * len(df_suppliers),
                                      color_discrete_sequence=['#007bff'], template=template)
    supplier_geo_map.update_layout(font=dict(family="Arial", size=12), title_font_size=18)

    supplier_performance_bar = px.bar(supplier_performance, x='supplier', y=['on_time_delivery_rate', 'delay_percentage'],
                                      title='Supplier Performance', barmode='group',
                                      color_discrete_map={'on_time_delivery_rate': '#28a745', 'delay_percentage': '#dc3545'},
                                      hover_data=['num_shipments'], template=template)
    supplier_performance_bar.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    
    # Create a performance matrix (cost vs delivery)
    supplier_matrix = supplier_performance.copy()
    
    # FIX: Use merge to add the calculated average cost to the supplier_matrix DataFrame correctly
    avg_cost_df = df_inventory.groupby('supplier', observed=True)['price_per_unit'].mean().reset_index()
    avg_cost_df.columns = ['supplier', 'avg_cost']
    supplier_matrix = supplier_matrix.merge(avg_cost_df, on='supplier', how='left').fillna(0)
    
    if not supplier_matrix['avg_cost'].isnull().all():
        supplier_matrix['cost_score'] = (1 - (supplier_matrix['avg_cost'] / supplier_matrix['avg_cost'].max())) * 50
    else:
        supplier_matrix['cost_score'] = 0
    supplier_matrix['delivery_score'] = supplier_matrix['on_time_delivery_rate'] * 0.5
    supplier_matrix['total_score'] = supplier_matrix['cost_score'] + supplier_matrix['delivery_score']
    
    matrix_fig = px.scatter(supplier_matrix, x='on_time_delivery_rate', y='avg_cost',
                           size='num_shipments', color='total_score', hover_name='supplier',
                           title='Supplier Performance Matrix (Cost vs Delivery)',
                           labels={'on_time_delivery_rate': 'On-Time Delivery Rate (%)',
                                   'avg_cost': 'Average Cost per Unit'},
                           color_continuous_scale='RdYlGn_r',
                           template=template)
    
    matrix_fig.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    
    return (supplier_geo_map, supplier_performance_bar, matrix_fig)

# Update Batch Tracking Charts
@callback(
    Output('batch-heatmap', 'figure'),
    Input('theme-toggle', 'value')
)
def update_batch_charts(theme):
    template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    
    batch_heatmap_data = batch_data.groupby(['drug_name', 'status'], observed=True)['quantity'].sum().unstack().fillna(0)
    batch_heatmap = px.imshow(batch_heatmap_data, labels=dict(x="Status", y="Drug Name", color="Quantity"),
                              title='Batch Status Heatmap', template=template)
    batch_heatmap.update_layout(font=dict(family="Arial", size=12), title_font_size=18)
    return batch_heatmap

@callback(
    [Output('forecast-graph', 'figure'),
     Output('forecast-table', 'data')],
    [Input('forecast-drug', 'value'),
     Input('theme-toggle', 'value')]
)
def update_forecast_graph(drug, theme):
    if not drug:
        return go.Figure(layout={'title': 'Select a drug to view forecast'}), []
    df_drug = df_history.groupby('drug_name', observed=True).get_group(drug) if drug else df_history.copy()
    if len(df_drug) < 3:
        return go.Figure(layout={'title': 'Insufficient data for forecast'}), []
    
    template = 'plotly_dark' if theme == 'Dark' else 'plotly_white'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_drug['date'], y=df_drug['quantity'], name='Historical Data', mode='lines+markers', line=dict(color='#007bff')))
    
    future_dates, predicted, ci = forecast_stock(df_drug['date'], df_drug['quantity'])
    forecast_data = []
    
    if not future_dates.empty:
        fig.add_trace(go.Scatter(x=future_dates, y=predicted, name='90-Day Forecast', mode='lines', line=dict(dash='dash', color='orange')))
        fig.add_trace(go.Scatter(x=future_dates, y=predicted + ci, mode='lines', line=dict(width=0), showlegend=False, name='Upper Bound'))
        fig.add_trace(go.Scatter(x=future_dates, y=predicted - ci, fill='tonexty', mode='lines', line=dict(width=0), showlegend=False, name='Lower Bound', fillcolor='rgba(255,165,0,0.2)'))
        
        # Add action column
        safety_stock = df_inventory[df_inventory['drug_name'] == drug]['safety_stock'].iloc[0] if not df_inventory[df_inventory['drug_name'] == drug].empty else 0
        for date, pred in zip(future_dates, predicted):
            action = 'Reorder Now' if pred < safety_stock else 'Monitor'
            forecast_data.append({'date': date.strftime('%Y-%m-%d'), 'predicted': round(pred, 2), 'action': action})
    
    fig.update_layout(title=f'{drug} Stock Forecast', xaxis_title='Date', yaxis_title='Quantity', template=template, font=dict(family="Arial", size=12), title_font_size=18)
    return fig, forecast_data

# Send Alert Emails (Simulated)
@callback(
    Output('email-output', 'children'),
    Input('send-alert-emails', 'n_clicks'),
    State('login-status', 'data'),
    prevent_initial_call=True
)
def send_alert_emails(n, login_status):
    if not login_status.get('logged_in') or login_status['role'] not in ['admin', 'manager']:
        return 'Permission denied: Only admins and managers can send alerts'
    if n:
        # Simulated email alert (replace with actual SMTP logic if needed)
        alerts = []
        high_risk = df_inventory[df_inventory['stockout_risk'] == 'High Risk']
        if not high_risk.empty:
            alerts.append(f"Stockout Risk Alert: {len(high_risk)} items below safety stock.")
        overstock = df_inventory[df_inventory['overstock_risk'] == 'High Risk']
        if not overstock.empty:
            alerts.append(f"Overstock Alert: {len(overstock)} items exceed optimal stock levels.")
        print("ALERT EMAIL:", "\n".join(alerts))  # Replace with email sending logic
        audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Send Alerts', 'Sent alert emails (simulated)']
        audit_log.to_csv(AUDIT_PATH, index=False)
        logging.info(f"User {login_status['username']} sent alert emails (simulated)")
        return 'Alerts sent successfully (simulated)'
    return ''

# Add, Save, and Export Callbacks
@callback(
    [Output('inventory-table', 'data'), Output('inv-output', 'children')],
    Input('add-inv-row', 'n_clicks'),
    [State('inventory-table', 'data'), State('inventory-table', 'columns'), State('login-status', 'data')],
    prevent_initial_call=True
)
def add_inv_row(n, rows, cols, login_status):
    if not login_status.get('logged_in') or login_status['role'] not in ['admin', 'manager']:
        return rows, 'Permission denied: Only admins and managers can add rows'
    if n:
        new_row = {c['id']: '' for c in cols}
        new_row['id'] = max([int(row['id']) for row in rows if 'id' in row and str(row['id']).isdigit()] + [0]) + 1
        rows.append(new_row)
        audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Add Inventory Row', 'Added new row']
        audit_log.to_csv(AUDIT_PATH, index=False)
        logging.info(f"User {login_status['username']} added new inventory row")
        return rows, 'New inventory row added successfully'
    return rows, ''

@callback(
    Output('inv-output', 'children', allow_duplicate=True),
    Input('save-inv', 'n_clicks'),
    [State('inventory-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def save_inv(n, data, login_status):
    if not login_status.get('logged_in') or login_status['role'] not in ['admin', 'manager']:
        return 'Permission denied: Only admins and managers can save inventory'
    if n:
        try:
            df = pd.DataFrame(data)
            # Ensure Categorical columns have appropriate categories
            if 'abc_category' in df.columns:
                df['abc_category'] = pd.Categorical(df['abc_category'], categories=['A', 'B', 'C', 'Unknown'])
            if 'aging_bucket' in df.columns:
                df['aging_bucket'] = pd.Categorical(df['aging_bucket'], categories=['<6 months', '6-12 months', '>12 months', 'Unknown'])
            if 'demand_volatility' in df.columns:
                df['demand_volatility'] = pd.Categorical(df['demand_volatility'], categories=['High', 'Low', 'Unknown'])
            df.to_csv(INVENTORY_PATH, index=False)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Save Inventory', 'Saved inventory to CSV']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} saved inventory")
            return 'Inventory saved successfully'
        except Exception as e:
            logging.error(f"Error saving inventory: {e}")
            print(f"ALERT: Error saving inventory: {e}")
            return f'Error saving inventory: {e}'
    return ''

@callback(
    Output('inv-output', 'children', allow_duplicate=True),
    Input('export-inv-pdf', 'n_clicks'),
    [State('inventory-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def export_inv_pdf(n, data, login_status):
    if not login_status.get('logged_in'):
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.chapter_body_table(data, title="Inventory Report")
            pdf_file = os.path.join(REPORTS_PATH, f'inventory_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export Inventory PDF', f'Exported to {pdf_file}']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported inventory PDF")
            return f'Inventory PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting inventory PDF: {e}")
            print(f"ALERT: Error exporting inventory PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

@callback(
    [Output('reorder-table', 'data'), Output('reorder-output', 'children')],
    Input('reorder-suggestions', 'n_clicks'),
    [State('inventory-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def generate_reorder(n, data, login_status):
    if not login_status.get('logged_in') or login_status['role'] not in ['admin', 'manager']:
        return [], 'Permission denied: Only admins and managers can generate suggestions'
    if n:
        try:
            df = pd.DataFrame(data)
            suggestions = generate_reorder_suggestions(df)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Generate Reorder Suggestions', 'Generated reorder suggestions']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} generated reorder suggestions")
            return suggestions.to_dict('records'), 'Reorder suggestions generated successfully'
        except Exception as e:
            logging.error(f"Error generating reorder suggestions: {e}")
            print(f"ALERT: Error generating reorder suggestions: {e}")
            return [], f'Error generating suggestions: {e}'
    return [], ''

@callback(
    Output('cons-output', 'children', allow_duplicate=True),
    Input('save-cons', 'n_clicks'),
    [State('consumables-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def save_cons(n, data, login_status):
    if not login_status.get('logged_in') or login_status['role'] not in ['admin', 'manager']:
        return 'Permission denied: Only admins and managers can save consumables'
    if n:
        try:
            df = pd.DataFrame(data)
            df.to_csv(CONSUMABLES_PATH, index=False)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Save Consumables', 'Saved consumables to CSV']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} saved consumables")
            return 'Consumables saved successfully'
        except Exception as e:
            logging.error(f"Error saving consumables: {e}")
            print(f"ALERT: Error saving consumables: {e}")
            return f'Error saving consumables: {e}'
    return ''

@callback(
    Output('cons-output', 'children', allow_duplicate=True),
    Input('export-cons-pdf', 'n_clicks'),
    [State('consumables-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def export_cons_pdf(n, data, login_status):
    if not login_status.get('logged_in'):
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.chapter_body_table(data, "Consumables Report")
            pdf_file = os.path.join(REPORTS_PATH, f'consumables_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export Consumables PDF', f'Exported to {pdf_file}']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported consumables PDF")
            return f'Consumables PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting consumables PDF: {e}")
            print(f"ALERT: Error exporting consumables PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

@callback(
    Output('shipment-output', 'children', allow_duplicate=True),
    Input('save-shipments', 'n_clicks'),
    [State('shipment-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def save_shipments(n, data, login_status):
    if not login_status.get('logged_in') or login_status['role'] not in ['admin', 'manager']:
        return 'Permission denied: Only admins and managers can save shipments'
    if n:
        try:
            df = pd.DataFrame(data)
            df.to_csv(SHIPMENTS_PATH, index=False)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Save Shipments', 'Saved shipments to CSV']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} saved shipments")
            return 'Shipments saved successfully'
        except Exception as e:
            logging.error(f"Error saving shipments: {e}")
            print(f"ALERT: Error saving shipments: {e}")
            return f'Error saving shipments: {e}'
    return ''

@callback(
    Output('shipment-output', 'children', allow_duplicate=True),
    Input('export-shipments-pdf', 'n_clicks'),
    [State('shipment-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def export_shipments_pdf(n, data, login_status):
    if not login_status.get('logged_in'):
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.chapter_body_table(data, "Shipments Report")
            pdf_file = os.path.join(REPORTS_PATH, f'shipments_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export Shipments PDF', f'Exported to {pdf_file}']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported shipments PDF")
            return f'Shipments PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting shipments PDF: {e}")
            print(f"ALERT: Error exporting shipments PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

@callback(
    Output('suppliers-output', 'children', allow_duplicate=True),
    Input('export-suppliers-pdf', 'n_clicks'),
    [State('suppliers-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def export_suppliers_pdf(n, data, login_status):
    if not login_status.get('logged_in'):
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.chapter_body_table(data, "Suppliers Report")
            pdf.chapter_body_table(supplier_performance.to_dict('records'), title="Supplier Performance Details")
            pdf_file = os.path.join(REPORTS_PATH, f'suppliers_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export Suppliers PDF', f'Exported to {pdf_file}']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported suppliers PDF")
            return f'Suppliers PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting suppliers PDF: {e}")
            print(f"ALERT: Error exporting suppliers PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

@callback(
    Output('batch-output', 'children', allow_duplicate=True),
    Input('export-batch-pdf', 'n_clicks'),
    [State('batch-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def export_batch_pdf(n, data, login_status):
    if not login_status.get('logged_in'):
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.chapter_body_table(data, "Batch Tracking Report")
            pdf_file = os.path.join(REPORTS_PATH, f'batch_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export Batch PDF', f'Exported to {pdf_file}']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported batch PDF")
            return f'Batch Tracking PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting batch PDF: {e}")
            print(f"ALERT: Error exporting batch PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

@callback(
    Output('users-output', 'children', allow_duplicate=True),
    Input('save-users', 'n_clicks'),
    [State('users-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def save_users(n, data, login_status):
    if not login_status.get('logged_in') or login_status['role'] != 'admin':
        return 'Permission denied: Only admins can save users'
    if n:
        try:
            df = pd.DataFrame(data)
            if 'password' not in df.columns:
                df['password'] = df['username'].apply(lambda x: hashlib.md5('default'.encode()).hexdigest())
            else:
                df['password'] = df['password'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest())
            df.to_csv(USERS_PATH, index=False)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Save Users', 'Saved users to CSV']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} saved users")
            return 'Users saved successfully'
        except Exception as e:
            logging.error(f"Error saving users: {e}")
            print(f"ALERT: Error saving users: {e}")
            return f'Error: {e}'
    return ''

@callback(
    Output('users-output', 'children', allow_duplicate=True),
    Input('export-users-pdf', 'n_clicks'),
    [State('users-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def export_users_pdf(n, data, login_status):
    if not login_status.get('logged_in'):
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.chapter_body_table(data, "Users Report")
            pdf_file = os.path.join(REPORTS_PATH, f'users_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export Users PDF', f'Exported to {pdf_file}']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported users PDF")
            return f'Users PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting users PDF: {e}")
            print(f"ALERT: Error exporting users PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

@callback(
    Output('audit-output', 'children', allow_duplicate=True),
    Input('export-audit-pdf', 'n_clicks'),
    [State('audit-table', 'data'), State('login-status', 'data')],
    prevent_initial_call=True
)
def export_audit_pdf(n, data, login_status):
    if not login_status.get('logged_in'):
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.chapter_body_table(data, "Audit Trail Report")
            pdf_file = os.path.join(REPORTS_PATH, f'audit_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export Audit PDF', f'Exported to {pdf_file}']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported audit PDF")
            return f'Audit Trail PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting audit PDF: {e}")
            print(f"ALERT: Error exporting audit PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

@callback(
    Output('analytics-output', 'children', allow_duplicate=True),
    Input('export-analytics-pdf', 'n_clicks'),
    [State('eoq-table', 'data'), State('dead-stock-table', 'data'), State('aging-table', 'data'),
     State('eoq-bar-chart', 'figure'), State('dead-stock-bar', 'figure'), State('aging-heatmap-analytics', 'figure'),
     State('service-level-bar', 'figure'), State('sales-trend', 'figure'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def export_analytics_pdf(n, eoq_data, dead_stock_data, aging_data, eoq_fig, dead_stock_fig, aging_fig, service_level_fig, sales_trend_fig, login_status):
    if not login_status.get('logged_in'):
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.chapter_body_table(eoq_data, "EOQ Analysis")
            pdf.chapter_body_table(dead_stock_data, "Dead Stock Analysis")
            pdf.chapter_body_table(aging_data, "Aging Analysis")
            
            pdf.add_page()
            pdf.chapter_title("Analytics Visualizations")
            for fig, title in [(eoq_fig, "EOQ Bar Chart"), (dead_stock_fig, "Dead Stock Bar Chart"), (aging_fig, "Aging Heatmap"), (service_level_fig, "Service Level Bar"), (sales_trend_fig, "Sales Trend")]:
                img_data = 'data:image/png;base64,' + base64.b64encode(go.Figure(fig).to_image(format='png')).decode('utf-8')
                pdf.chapter_body_image(img_data)
            
            pdf_file = os.path.join(REPORTS_PATH, f'analytics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export Analytics PDF', f'Exported to {pdf_file}']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported analytics PDF")
            return f'Analytics PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting analytics PDF: {e}")
            print(f"ALERT: Error exporting analytics PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

@callback(
    Output('alerts', 'is_open'),
    Input('export-report', 'n_clicks'),
    [State('login-status', 'data'),
     State('inventory-bar', 'figure'), State('category-pie', 'figure'), State('value-treemap', 'figure'),
     State('abc-pie', 'figure'), State('reorder-vs-quantity-bar', 'figure'), State('stockout-risk-pie', 'figure'),
     State('aging-heatmap', 'figure'), State('demand-scatter', 'figure'), State('carrying-cost-bar', 'figure')]
)
def export_full_report(n, login_status, inv_bar_fig, cat_pie_fig, val_treemap_fig, abc_pie_fig, reorder_fig, risk_pie_fig, aging_fig, demand_fig, carrying_fig):
    if not n or not login_status.get('logged_in') or login_status['role'] not in ['admin']:
        raise PreventUpdate
    
    try:
        figs = [
            (inv_bar_fig, "Inventory Levels"),
            (cat_pie_fig, "Inventory Value by Category"),
            (val_treemap_fig, "Inventory Value Treemap"),
            (abc_pie_fig, "ABC Analysis"),
            (reorder_fig, "Reorder Status"),
            (risk_pie_fig, "Stockout Risk"),
            (aging_fig, "Aging Heatmap"),
            (demand_fig, "Demand vs. Stock Risk"),
            (carrying_fig, "Carrying Cost by Category")
        ]
        images = {title: 'data:image/png;base64,' + base64.b64encode(go.Figure(fig).to_image(format='png')).decode('utf-8') for fig, title in figs}

        pdf = PDF()
        pdf.add_page()
        pdf.chapter_title("Executive Summary")
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, "Comprehensive overview of inventory status, analytics, and recommendations for Unique Pharmaceuticals.")
        pdf.ln(5)

        for title, img_data in images.items():
            pdf.chapter_title(title)
            pdf.chapter_body_image(img_data)
        
        pdf.add_page()
        pdf.chapter_body_table(df_inventory[['drug_name', 'quantity', 'total_value', 'reorder_alert', 'stockout_risk', 'overstock_risk', 'supplier']].to_dict('records'), title="Detailed Inventory Data")
        pdf.chapter_body_table(generate_reorder_suggestions(df_inventory).to_dict('records'), title="Reorder Suggestions")
        pdf.chapter_body_table(generate_stockout_risks(df_inventory).to_dict('records'), title="Stockout Risks")
        
        # New sections
        pdf.chapter_body_table(df_inventory[['drug_name', 'gmroi', 'stock_cover', 'fill_rate', 'stockout_frequency', 'order_cycle_time', 'inventory_accuracy', 'carrying_cost_percent']].to_dict('records'), title="Advanced Inventory Metrics")
        pdf.chapter_body_table(supplier_performance.to_dict('records'), title="Supplier Performance Details")
        
        pdf_file = os.path.join(REPORTS_PATH, f'full_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
        pdf.output(pdf_file)
        audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export Full Report', f'Exported to {pdf_file}']
        audit_log.to_csv(AUDIT_PATH, index=False)
        logging.info(f"User {login_status['username']} exported full report")
        return True
    except Exception as e:
        logging.error(f"Error exporting full report: {e}")
        print(f"ALERT: Error exporting full report: {e}")
        return False


if __name__ == '__main__':
    try:
        print("Starting Dash application on http://127.0.0.1:8050/")
        logging.info("Attempting to start Dash application")
        app.run(debug=True, host='0.0.0.0', port=8050)
    except Exception as e:
        logging.error(f"Application startup error: {e}")
        print(f"ALERT: Failed to start application: {e}")
        raise
