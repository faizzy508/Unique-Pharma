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

# File paths (moved before logging)
BASE_PATH = os.environ.get('BASE_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
INVENTORY_PATH = os.path.join(BASE_PATH, "inventory.csv")
CONSUMABLES_PATH = os.path.join(BASE_PATH, "consumables.csv")
HISTORY_PATH = os.path.join(BASE_PATH, "history.csv")
SUPPLIERS_PATH = os.path.join(BASE_PATH, "suppliers.csv")
USERS_PATH = os.path.join(BASE_PATH, "users.csv")
AUDIT_PATH = os.path.join(BASE_PATH, "audit_log.csv")
SHIPMENTS_PATH = os.path.join(BASE_PATH, "shipments.csv")
REPORTS_PATH = os.path.join(BASE_PATH, "reports")

# Create BASE_PATH directory if it doesn't exist (fixes FileNotFoundError for log and other files)
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(BASE_PATH, "dashboard.log"),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create reports directory if it doesn't exist
if not os.path.exists(REPORTS_PATH):
    os.makedirs(REPORTS_PATH)

# Dash app initialization
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.FLATLY,
    "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css",
    "/assets/custom.css"
], suppress_callback_exceptions=True)
server = app.server  # Required for deployment with Gunicorn

# Load data with error handling
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
    raise Exception(f"Missing CSV file: {e}")
except Exception as e:
    logging.error(f"Data loading error: {e}")
    raise Exception(f"Error loading data: {e}")

# Process data with correct date parsing
current_date = datetime.now()  # Updated to current date and time
try:
    df_inventory['expiration_date'] = pd.to_datetime(df_inventory['expiration_date'], format='%d/%m/%Y', errors='coerce')
    df_inventory['expired'] = df_inventory['expiration_date'].notna() & (df_inventory['expiration_date'] < current_date)
    df_inventory['quantity'] = pd.to_numeric(df_inventory['quantity'], errors='coerce').fillna(0)
    df_inventory['price_per_unit'] = pd.to_numeric(df_inventory['price_per_unit'], errors='coerce').fillna(0)
    df_inventory['total_value'] = df_inventory['quantity'] * df_inventory['price_per_unit']
    df_inventory['lead_time_days'] = pd.to_numeric(df_inventory['lead_time_days'], errors='coerce').fillna(180)
    df_inventory['average_daily_demand'] = pd.to_numeric(df_inventory['average_daily_demand'], errors='coerce').fillna(0)
    df_inventory['std_dev_demand'] = pd.to_numeric(df_inventory['std_dev_demand'], errors='coerce').fillna(0)
    df_inventory['lead_time_demand'] = df_inventory['average_daily_demand'] * df_inventory['lead_time_days']
    df_inventory['safety_stock'] = 1.65 * np.sqrt(df_inventory['lead_time_days']) * df_inventory['std_dev_demand']
    df_inventory['reorder_point'] = df_inventory['lead_time_demand'] + df_inventory['safety_stock']
    df_inventory['minimum_stock'] = df_inventory['average_daily_demand'] * 180
    df_inventory['reorder_alert'] = np.where(df_inventory['quantity'] < df_inventory['reorder_point'], 'Reorder Now', 'Sufficient')

    # ABC Analysis
    total_value_sum = df_inventory['total_value'].sum()
    df_inventory = df_inventory.sort_values('total_value', ascending=False)
    df_inventory['cumulative_value'] = df_inventory['total_value'].cumsum() / total_value_sum
    df_inventory['abc_category'] = np.where(df_inventory['cumulative_value'] <= 0.8, 'A',
                                            np.where(df_inventory['cumulative_value'] <= 0.95, 'B', 'C'))

    df_consumables['quantity'] = pd.to_numeric(df_consumables['quantity'], errors='coerce').fillna(0)
    df_consumables['price_per_unit'] = pd.to_numeric(df_consumables['price_per_unit'], errors='coerce').fillna(0)
    df_consumables['total_cost'] = df_consumables['quantity'] * df_consumables['price_per_unit']
    df_consumables['lead_time_days'] = pd.to_numeric(df_consumables['lead_time_days'], errors='coerce').fillna(180)
    df_consumables['average_daily_demand'] = pd.to_numeric(df_consumables['average_daily_demand'], errors='coerce').fillna(0)
    df_consumables['std_dev_demand'] = pd.to_numeric(df_consumables['std_dev_demand'], errors='coerce').fillna(0)
    df_consumables['lead_time_demand'] = df_consumables['average_daily_demand'] * df_consumables['lead_time_days']
    df_consumables['safety_stock'] = 1.65 * np.sqrt(df_consumables['lead_time_days']) * df_consumables['std_dev_demand']
    df_consumables['reorder_point'] = df_consumables['lead_time_demand'] + df_consumables['safety_stock']
    df_consumables['minimum_stock'] = df_consumables['average_daily_demand'] * 180
    df_consumables['reorder_alert'] = np.where(df_consumables['quantity'] < df_consumables['reorder_point'], 'Reorder Now', 'Sufficient')
    df_history['quantity'] = pd.to_numeric(df_history['quantity'], errors='coerce').fillna(0)
    df_history['date'] = pd.to_datetime(df_history['date'], format='%Y-%m-%d', errors='coerce')
    df_shipments['order_date'] = pd.to_datetime(df_shipments['order_date'], errors='coerce')
    df_shipments['expected_arrival'] = pd.to_datetime(df_shipments['expected_arrival'], errors='coerce')
    df_inv_with_start = df_inventory.copy()
    df_inv_with_start['x_start'] = current_date
except Exception as e:
    logging.error(f"Data processing error: {e}")
    raise Exception(f"Error processing data: {e}")

# Low stock threshold
LOW_STOCK_THRESHOLD = 1000
df_inventory['low_stock'] = df_inventory['quantity'] < LOW_STOCK_THRESHOLD

# Additional calculations for stockout risk
df_inventory['stockout_risk'] = np.where(df_inventory['quantity'] < df_inventory['safety_stock'], 'High Risk', 'Low Risk')

# Inventory Turnover Calculation
def calculate_inventory_turnover():
    turnover_data = []
    for drug in df_inventory['drug_name'].unique():
        hist_data = df_history[df_history['drug_name'] == drug].sort_values('date')
        if not hist_data.empty:
            initial_qty = hist_data.iloc[0]['quantity']
            final_qty = df_inventory[df_inventory['drug_name'] == drug]['quantity'].iloc[0]
            total_sold = initial_qty - final_qty if initial_qty > final_qty else 0
            avg_stock = (hist_data['quantity'].mean() + final_qty) / 2
            turnover_rate = total_sold / avg_stock if avg_stock > 0 else 0
            turnover_data.append({
                'drug_name': drug,
                'total_sold': total_sold,
                'average_stock': round(avg_stock, 2),
                'turnover_rate': round(turnover_rate, 2)
            })
    return pd.DataFrame(turnover_data)

# Calculate turnover_df before KPIs
turnover_df = calculate_inventory_turnover()

# KPIs
total_inventory_value = df_inventory['total_value'].sum()
num_expired = df_inventory['expired'].sum()
num_low_stock = df_inventory['low_stock'].sum()
total_consumables_cost = df_consumables['total_cost'].sum()
num_a_items = df_inventory[df_inventory['abc_category'] == 'A'].shape[0]
a_value_percent = df_inventory[df_inventory['abc_category'] == 'A']['total_value'].sum() / total_inventory_value * 100 if total_inventory_value > 0 else 0
avg_turnover_rate = turnover_df['turnover_rate'].mean() if not turnover_df.empty else 0
num_delayed_shipments = df_shipments[df_shipments['status'] == 'Delayed'].shape[0]
num_stockout_risk = df_inventory[df_inventory['stockout_risk'] == 'High Risk'].shape[0]

# Supplier locations
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

# Placeholder for batch_data (define or load as needed; assuming empty for now)
batch_data = pd.DataFrame(columns=['batch_number', 'drug_name', 'quantity', 'expiration_date', 'status'])

# Supplier performance metrics
supplier_performance = df_shipments.groupby('supplier').agg({
    'quantity': 'count',
    'status': lambda x: (x == 'Delayed').mean() * 100
}).reset_index()
supplier_performance.columns = ['supplier', 'num_shipments', 'delay_percentage']
supplier_performance['on_time_delivery_rate'] = 100 - supplier_performance['delay_percentage']

# Forecasting function
def forecast_stock(dates, quantities):
    try:
        dates_num = pd.to_datetime(dates).map(datetime.toordinal).values.reshape(-1, 1)
        quantities = pd.to_numeric(quantities, errors='coerce').values.reshape(-1, 1)
        if len(dates_num) < 2 or np.all(np.isnan(quantities)):
            return pd.DatetimeIndex([]), np.array([]), 0
        model = LinearRegression()
        model.fit(dates_num, quantities)
        future_dates = pd.date_range(dates.min(), dates.max() + pd.Timedelta(days=90), freq='D')
        future_dates_num = future_dates.map(datetime.toordinal).values.reshape(-1, 1)
        predicted = model.predict(future_dates_num)
        y_pred = model.predict(dates_num)
        mse = np.mean((y_pred - quantities) ** 2, where=~np.isnan(quantities))
        std_err = np.sqrt(mse) if mse > 0 else 0
        ci = 1.96 * std_err
        return future_dates, predicted.flatten(), ci
    except Exception as e:
        logging.error(f"Forecasting error: {e}")
        return pd.DatetimeIndex([]), np.array([]), 0

# Reorder suggestions function
def generate_reorder_suggestions(df):
    suggestions = df[df['reorder_alert'] == 'Reorder Now'].copy()
    suggestions['suggested_order_quantity'] = (suggestions['reorder_point'] - suggestions['quantity'] + suggestions['safety_stock']).round()
    suggestions['estimated_cost'] = suggestions['suggested_order_quantity'] * suggestions['price_per_unit']
    return suggestions[['drug_name', 'quantity', 'reorder_point', 'safety_stock', 'suggested_order_quantity', 'estimated_cost', 'supplier']]

# Stockout risk table
def generate_stockout_risks(df):
    risks = df[df['stockout_risk'] == 'High Risk'].copy()
    return risks[['drug_name', 'quantity', 'safety_stock', 'reorder_point', 'stockout_risk']]

# Batch tracking data
batch_data = df_inventory[['batch_number', 'drug_name', 'quantity', 'expiration_date', 'supplier']].copy()
batch_data['status'] = np.where(batch_data['expiration_date'] < current_date, 'Expired', 'Active')

# Additional Inventory Analysis: Stock Age
df_inventory['stock_age_days'] = (current_date - df_inventory['expiration_date']).dt.days.abs() if 'expiration_date' in df_inventory else 0
stock_age_df = df_inventory[['drug_name', 'stock_age_days', 'quantity']].copy()

# Additional Inventory Analysis: Demand Variability
demand_variability_df = df_inventory[['drug_name', 'average_daily_demand', 'std_dev_demand']].copy()
demand_variability_df['variability_index'] = np.where(
    demand_variability_df['average_daily_demand'] != 0,
    demand_variability_df['std_dev_demand'] / demand_variability_df['average_daily_demand'],
    0
)

# Additional Inventory Analysis: Stock Value Trend
stock_value_trend_df = df_inventory[['drug_name', 'total_value', 'quantity']].copy()
stock_value_trend_df['value_per_unit'] = stock_value_trend_df['total_value'] / stock_value_trend_df['quantity'] if stock_value_trend_df['quantity'].any() else 0

# Custom CSS for business professional theme
custom_css = """
body {
    font-family: 'Arial', sans-serif;
    font-size: 14px;
    background-color: #f8f9fa;
    color: #333333;
}
.card {
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transition: box-shadow 0.3s;
    background-color: #ffffff;
}
.card:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}
.btn {
    border-radius: 4px;
    transition: background-color 0.3s;
}
.navbar {
    background-color: #ffffff;
    border-bottom: 1px solid #dee2e6;
}
.tab-content {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #dee2e6;
}
.data-table {
    font-size: 14px;
    color: #333333;
}
.data-table th {
    background-color: #e9ecef;
    color: #495057;
    font-weight: bold;
    padding: 10px;
    text-align: center;
}
.data-table td {
    padding: 10px;
    text-align: center;
    border-top: 1px solid #dee2e6;
}
.number {
    font-family: 'Arial', sans-serif;
    font-size: 16px;
    font-weight: bold;
    color: #007bff;
}
"""

# Save custom CSS to a local file
custom_css_file = os.path.join(BASE_PATH, "custom.css")
with open(custom_css_file, "w") as f:
    f.write(custom_css)

# Initialize Dash app with Flatly theme
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.FLATLY,
    "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css",
    "/assets/custom.css"
], suppress_callback_exceptions=True)

# PDF export class
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Unique Pharmaceuticals Inventory Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} | Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by Unique Pharmaceuticals Dashboard', 0, 0, 'C')

# Login layout
# Login layout (updated)
login_layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Img(src="/assets/Uniquesarl_Logo.png", height="120px", className="mb-4 mx-auto d-block animate__animated animate__fadeIn"),
                html.H2("Unique Pharmaceuticals Dashboard", className="text-center mb-4 animate__animated animate__fadeIn", style={'color': '#007bff', 'fontWeight': 'bold'}),
                dcc.Input(id='username', type='text', placeholder='Username', className='form-control mb-3', style={'borderRadius': '4px'}),
                dcc.Input(id='password', type='password', placeholder='Password', className='form-control mb-3', style={'borderRadius': '4px'}),
                dcc.Loading(
                    type="circle",
                    children=dbc.Button('Login', id='login-button', color='primary', className='btn-block mb-2 animate__animated animate__pulse')
                ),
                html.Div(id='login-output', className='text-danger mt-2 text-center')
            ], width=4, className='mx-auto shadow p-5 bg-white rounded')
        ], justify='center', className='min-vh-100 align-items-center')
    ], fluid=True, style={'background-color': '#f8f9fa'})
])

# Dashboard layout (updated)
dashboard_layout = html.Div([
    dcc.Interval(id='interval-update', interval=60*1000, n_intervals=0),
    dbc.Navbar(
        children=[
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/Uniquesarl_Logo.png", height="40px")),
                        dbc.Col(dbc.NavbarBrand("Unique Pharmaceuticals", className="ml-2", style={'color': '#495057', 'fontWeight': 'bold'})),
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
                        dbc.NavItem(dbc.NavLink(id='user-greeting', style={'color': '#495057'})),
                        dbc.NavItem(dbc.NavLink("Logout", id='logout-button', href="#", style={'color': '#495057'})),
                        dbc.NavItem(dcc.Dropdown(
                            ['Light', 'Dark'], 'Light', id='theme-toggle', clearable=False, style={'width': '120px', 'backgroundColor': '#ffffff', 'color': '#495057'}
                        )),
                        dbc.NavItem(dcc.Loading(
                            type="circle",
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
        sticky="top",
        style={'border-bottom': '1px solid #dee2e6'}
    ),
    dbc.Container(fluid=True, children=[
        html.H1("Pharmaceutical Management Dashboard", className="text-center my-4 animate__animated animate__fadeIn", style={'color': '#007bff', 'fontWeight': 'bold'}),
        dcc.Tabs(id="tabs", value='tab-executive', children=[
            dcc.Tab(label='Executive Overview', value='tab-executive', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("Executive Insights", className="text-center mb-4", style={'color': '#007bff'}),
                    dcc.Loading(
                        type="circle",
                        children=dbc.Alert(id='alerts', is_open=False, dismissable=True, color='warning', className="animate__animated animate__fadeIn")
                    ),
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(id='kpi-total-value', className="card-title animate__animated animate__zoomIn number"),
                                html.P("Total Inventory Value", className="card-text")
                            ])
                        ], color="primary", outline=True, id='kpi-total-value-card'), width=12, sm=6, md=3),
                        dbc.Tooltip("Sum of quantity * price per unit for all inventory items.", target='kpi-total-value-card'),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(id='kpi-expired', className="card-title animate__animated animate__zoomIn number"),
                                html.P("Expired Items", className="card-text")
                            ])
                        ], color="danger", outline=True, id='kpi-expired-card'), width=12, sm=6, md=3),
                        dbc.Tooltip("Number of items past their expiration date.", target='kpi-expired-card'),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(id='kpi-low-stock', className="card-title animate__animated animate__zoomIn number"),
                                html.P("Low Stock Items", className="card-text")
                            ])
                        ], color="warning", outline=True, id='kpi-low-stock-card'), width=12, sm=6, md=3),
                        dbc.Tooltip(f"Items with quantity below {LOW_STOCK_THRESHOLD}.", target='kpi-low-stock-card'),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(id='kpi-consumables-cost', className="card-title animate__animated animate__zoomIn number"),
                                html.P("Total Consumables Cost", className="card-text")
                            ])
                        ], color="success", outline=True, id='kpi-consumables-cost-card'), width=12, sm=6, md=3),
                        dbc.Tooltip("Total cost of all consumables.", target='kpi-consumables-cost-card'),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(id='kpi-a-items', className="card-title animate__animated animate__zoomIn number"),
                                html.P("A-Class Items (ABC)", className="card-text")
                            ])
                        ], color="info", outline=True, id='kpi-a-items-card'), width=12, sm=6, md=3),
                        dbc.Tooltip("Number of high-value A-class items (80% of value).", target='kpi-a-items-card'),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(id='kpi-avg-turnover', className="card-title animate__animated animate__zoomIn number"),
                                html.P("Avg Turnover Rate", className="card-text")
                            ])
                        ], color="secondary", outline=True, id='kpi-avg-turnover-card'), width=12, sm=6, md=3),
                        dbc.Tooltip("Average inventory turnover rate across all drugs.", target='kpi-avg-turnover-card'),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(id='kpi-delayed-shipments', className="card-title animate__animated animate__zoomIn number"),
                                html.P("Delayed Shipments", className="card-text")
                            ])
                        ], color="danger", outline=True, id='kpi-delayed-shipments-card'), width=12, sm=6, md=3),
                        dbc.Tooltip("Number of delayed shipments.", target='kpi-delayed-shipments-card'),
                        dbc.Col(dbc.Card([
                            dbc.CardBody([
                                html.H4(id='kpi-stockout-risk', className="card-title animate__animated animate__zoomIn number"),
                                html.P("Stockout Risk Items", className="card-text")
                            ])
                        ], color="warning", outline=True, id='kpi-stockout-risk-card'), width=12, sm=6, md=3),
                        dbc.Tooltip("Number of items at high stockout risk (quantity < safety stock).", target='kpi-stockout-risk-card'),
                    ], className="mb-4", justify="center"),
                    html.H4("Key Insights", className="text-center mb-3", style={'color': '#007bff'}),
                    html.Div(id='insights-text', className="mb-4 p-3 bg-light border rounded animate__animated animate__fadeIn shadow-sm"),
                    dcc.Loading(
                        type="circle",
                        children=dbc.Button('Generate AI Insights', id='ai-insights-button', color='warning', className='mb-3')
                    ),
                    dcc.Loading(
                        type="circle",
                        children=dbc.Button('Send Alert Emails', id='send-alert-emails', color='danger', className='mb-3 ml-2')
                    ),
                    html.Div(id='email-output', className='text-success mt-2 text-center'),
                    dcc.Dropdown(
                        id='category-filter',
                        options=[{'label': cat, 'value': cat} for cat in df_inventory['category'].unique()],
                        multi=True,
                        placeholder="Filter by Category",
                        className="mb-3",
                        style={'backgroundColor': '#ffffff', 'color': '#495057', 'borderRadius': '4px'}
                    ),
                    dcc.Dropdown(
                        id='drug-filter',
                        options=[{'label': drug, 'value': drug} for drug in df_inventory['drug_name'].unique()],
                        multi=True,
                        placeholder="Filter by Drug",
                        className="mb-3",
                        style={'backgroundColor': '#ffffff', 'color': '#495057', 'borderRadius': '4px'}
                    ),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='inventory-bar'), width=12, md=6),
                        dbc.Col(dcc.Graph(id='category-pie'), width=12, md=6),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='executive-supplier-map'), width=12, md=6),
                        dbc.Col(dcc.Graph(id='consumables-bar'), width=12, md=6),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='history-line'), width=12, md=6),
                        dbc.Col(dcc.Graph(id='price-quantity-scatter'), width=12, md=6),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='quantity-heatmap'), width=12, md=6),
                        dbc.Col(dcc.Graph(id='expiration-timeline'), width=12, md=6),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='supplier-count-bar'), width=12, md=6),
                        dbc.Col(dcc.Graph(id='value-treemap'), width=12, md=6),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='quantity-box'), width=12, md=6),
                        dbc.Col(dcc.Graph(id='consumables-cost-pie'), width=12, md=6),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='abc-pie'), width=12, md=6),
                        dbc.Col(dcc.Graph(id='shipment-timeline'), width=12, md=6),
                    ], className="mb-4"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='reorder-vs-quantity-bar'), width=12, md=6),
                        dbc.Col(dcc.Graph(id='stockout-risk-pie'), width=12, md=6),
                    ], className="mb-4"),
                ]), className="shadow mb-5 bg-white rounded p-4")
            ]),
            dcc.Tab(label='Inventory Management', value='tab-inventory', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("Inventory Management", className="text-center mb-4", style={'color': '#007bff'}),
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
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'expired', 'filter_query': '{expired} eq true'},
                             'backgroundColor': '#f8d7da', 'color': '#721c24'},
                            {'if': {'column_id': 'low_stock', 'filter_query': '{low_stock} eq true'},
                             'backgroundColor': '#fff3cd', 'color': '#856404'},
                            {'if': {'column_id': 'reorder_alert', 'filter_query': '{reorder_alert} eq "Reorder Now"'},
                             'backgroundColor': '#ffdab9', 'color': '#804000'},
                            {'if': {'column_id': 'abc_category', 'filter_query': '{abc_category} eq "A"'},
                             'backgroundColor': '#d4edda', 'color': '#155724'},
                            {'if': {'column_id': 'stockout_risk', 'filter_query': '{stockout_risk} eq "High Risk"'},
                             'backgroundColor': '#f8d7da', 'color': '#721c24'}
                        ]
                    ),
                    dbc.Row([
                        dbc.Col(dbc.Button('Add Row', id='add-inv-row', color='primary', className='mr-2')),
                        dbc.Col(dbc.Button('Save to CSV', id='save-inv', color='success', className='mr-2')),
                        dbc.Col(dcc.Loading(
                            type="circle",
                            children=dbc.Button('Export to PDF', id='export-inv-pdf', color='info', className='mr-2')
                        )),
                        dbc.Col(dcc.Loading(
                            type="circle",
                            children=dbc.Button('Generate Reorder Suggestions', id='reorder-suggestions', color='warning')
                        )),
                    ], className="my-3", justify="center"),
                    html.Div(id='inv-output', className='text-success mt-2 text-center'),
                    html.H4("Reorder Suggestions", className="text-center mt-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='reorder-table',
                        columns=[
                            {"name": "Drug Name", "id": "drug_name"},
                            {"name": "Current Quantity", "id": "quantity"},
                            {"name": "Reorder Point", "id": "reorder_point"},
                            {"name": "Safety Stock", "id": "safety_stock"},
                            {"name": "Suggested Order Qty", "id": "suggested_order_quantity"},
                            {"name": "Estimated Cost", "id": "estimated_cost"},
                            {"name": "Supplier", "id": "supplier"}
                        ],
                        data=[],
                        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'suggested_order_quantity'}, 'backgroundColor': '#fff3cd', 'color': '#856404'}
                        ]
                    ),
                    html.Div(id='reorder-output', className='text-success mt-2 text-center'),
                    html.H4("Stockout Risks", className="text-center mt-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='stockout-risk-table',
                        columns=[
                            {"name": "Drug Name", "id": "drug_name"},
                            {"name": "Current Quantity", "id": "quantity"},
                            {"name": "Safety Stock", "id": "safety_stock"},
                            {"name": "Reorder Point", "id": "reorder_point"},
                            {"name": "Stockout Risk", "id": "stockout_risk"}
                        ],
                        data=[],
                        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'stockout_risk', 'filter_query': '{stockout_risk} eq "High Risk"'},
                             'backgroundColor': '#f8d7da', 'color': '#721c24'}
                        ]
                    ),
                    html.Div(id='stockout-output', className='text-success mt-2 text-center')
                ]), className="shadow mb-5 bg-white rounded p-4")
            ]),
            dcc.Tab(label='Consumables Management', value='tab-consumables', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("Consumables Management", className="text-center mb-4", style={'color': '#007bff'}),
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
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'reorder_alert', 'filter_query': '{reorder_alert} eq "Reorder Now"'},
                             'backgroundColor': '#ffdab9', 'color': '#804000'}
                        ]
                    ),
                    dbc.Row([
                        dbc.Col(dbc.Button('Add Row', id='add-cons-row', color='primary', className='mr-2')),
                        dbc.Col(dbc.Button('Save to CSV', id='save-cons', color='success', className='mr-2')),
                        dbc.Col(dcc.Loading(
                            type="circle",
                            children=dbc.Button('Export to PDF', id='export-cons-pdf', color='info')
                        )),
                    ], className="my-3", justify="center"),
                    html.Div(id='cons-output', className='text-success mt-2 text-center')
                ]), className="shadow mb-5 bg-white rounded p-4")
            ]),
            dcc.Tab(label='Shipment Tracking', value='tab-shipments', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("Shipment Tracking", className="text-center mb-4", style={'color': '#007bff'}),
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
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'status', 'filter_query': '{status} eq "Delayed"'},
                             'backgroundColor': '#f8d7da', 'color': '#721c24'},
                            {'if': {'column_id': 'status', 'filter_query': '{status} eq "In Transit"'},
                             'backgroundColor': '#ffdab9', 'color': '#804000'}
                        ]
                    ),
                    dbc.Row([
                        dbc.Col(dbc.Button('Add Shipment', id='add-shipment-row', color='primary', className='mr-2')),
                        dbc.Col(dbc.Button('Save Shipments', id='save-shipments', color='success', className='mr-2')),
                        dbc.Col(dcc.Loading(
                            type="circle",
                            children=dbc.Button('Export to PDF', id='export-shipments-pdf', color='info')
                        )),
                    ], className="my-3", justify="center"),
                    html.Div(id='shipment-output', className='text-success mt-2 text-center')
                ]), className="shadow mb-5 bg-white rounded p-4")
            ]),
            dcc.Tab(label='Inventory Metrics', value='tab-metrics', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("Inventory Metrics", className="text-center mb-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='inventory-metrics-table',
                        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in ['drug_name', 'lead_time_demand', 'safety_stock', 'reorder_point', 'minimum_stock', 'quantity', 'reorder_alert', 'abc_category', 'stockout_risk']],
                        data=df_inventory.to_dict('records'),
                        filter_action='native',
                        sort_action='native',
                        page_size=10,
                        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'reorder_alert', 'filter_query': '{reorder_alert} eq "Reorder Now"'},
                             'backgroundColor': '#ffdab9', 'color': '#804000'},
                            {'if': {'column_id': 'abc_category', 'filter_query': '{abc_category} eq "A"'},
                             'backgroundColor': '#d4edda', 'color': '#155724'},
                            {'if': {'column_id': 'stockout_risk', 'filter_query': '{stockout_risk} eq "High Risk"'},
                             'backgroundColor': '#f8d7da', 'color': '#721c24'}
                        ]
                    ),
                    dcc.Loading(
                        type="circle",
                        children=dbc.Button('Export to PDF', id='export-metrics-pdf', color='info', className='mt-3')
                    ),
                    html.Div(id='metrics-output', className='text-success mt-2 text-center')
                ]), className="shadow mb-5 bg-white rounded p-4")
            ]),
            dcc.Tab(label='Forecasting', value='tab-forecasting', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("Stock Forecasting", className="text-center mb-4", style={'color': '#007bff'}),
                    dcc.Dropdown(
                        id='forecast-drug',
                        options=[{'label': drug, 'value': drug} for drug in df_history['drug_name'].unique()],
                        placeholder="Select Drug for Forecasting",
                        className="mb-3",
                        style={'backgroundColor': '#ffffff', 'color': '#495057', 'borderRadius': '4px'}
                    ),
                    dcc.Graph(id='forecast-graph'),
                    html.H4("Forecast Values", className="text-center mt-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='forecast-table',
                        columns=[
                            {"name": "Date", "id": "date"},
                            {"name": "Predicted Quantity", "id": "predicted"}
                        ],
                        data=[],
                        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'}
                    ),
                ]), className="shadow mb-5 bg-white rounded p-4")
            ]),
            dcc.Tab(label='Suppliers', value='tab-suppliers', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("Supplier Management", className="text-center mb-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='suppliers-table',
                        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in df_suppliers.columns if i not in ['lat', 'lon', 'location_name']],
                        data=df_suppliers.to_dict('records'),
                        editable=True,
                        row_deletable=True,
                        filter_action='native',
                        sort_action='native',
                        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'}
                    ),
                    dcc.Graph(id='supplier-geo-map'),
                    html.H4("Supplier Performance Metrics", className="text-center mt-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='supplier-performance-table',
                        columns=[
                            {"name": "Supplier", "id": "supplier"},
                            {"name": "Number of Shipments", "id": "num_shipments"},
                            {"name": "Delay Percentage (%)", "id": "delay_percentage"},
                            {"name": "On-Time Delivery Rate (%)", "id": "on_time_delivery_rate"}
                        ],
                        data=supplier_performance.to_dict('records'),
                        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'delay_percentage', 'filter_query': '{delay_percentage} > 50'},
                             'backgroundColor': '#f8d7da', 'color': '#721c24'},
                            {'if': {'column_id': 'on_time_delivery_rate', 'filter_query': '{on_time_delivery_rate} < 50'},
                             'backgroundColor': '#f8d7da', 'color': '#721c24'}
                        ]
                    ),
                    dcc.Graph(id='supplier-performance-bar'),
                    dcc.Loading(
                        type="circle",
                        children=dbc.Button('Export to PDF', id='export-suppliers-pdf', color='info', className='mt-3')
                    ),
                    html.Div(id='suppliers-output', className='text-success mt-2 text-center')
                ]), className="shadow mb-5 bg-white rounded p-4")
            ]),
            dcc.Tab(label='Batch Tracking', value='tab-batch', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("Batch Tracking", className="text-center mb-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='batch-table',
                        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in batch_data.columns],
                        data=batch_data.to_dict('records'),
                        filter_action='native',
                        sort_action='native',
                        page_size=10,
                        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'status', 'filter_query': '{status} eq "Expired"'},
                             'backgroundColor': '#f8d7da', 'color': '#721c24'}
                        ]
                    ),
                    dcc.Graph(id='batch-heatmap'),
                    dcc.Loading(
                        type="circle",
                        children=dbc.Button('Export to PDF', id='export-batch-pdf', color='info', className='mt-3')
                    ),
                    html.Div(id='batch-output', className='text-success mt-2 text-center')
                ]), className="shadow mb-5 bg-white rounded p-4")
            ]),
            dcc.Tab(label='Inventory Analytics', value='tab-analytics', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("Inventory Analytics", className="text-center mb-4", style={'color': '#007bff'}),
                    html.H4("Turnover Metrics", className="text-center mt-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='analytics-table',
                        columns=[
                            {"name": "Drug Name", "id": "drug_name"},
                            {"name": "Total Sold", "id": "total_sold"},
                            {"name": "Average Stock", "id": "average_stock"},
                            {"name": "Turnover Rate", "id": "turnover_rate"}
                        ],
                        data=turnover_df.to_dict('records'),
                        filter_action='native',
                        sort_action='native',
                        page_size=10,
                        style_table={'overflowX': 'auto', 'borderRadius': '8px', 'className': 'data-table'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'turnover_rate', 'filter_query': '{turnover_rate} > 2'},
                             'backgroundColor': '#d4edda', 'color': '#155724'},
                            {'if': {'column_id': 'turnover_rate', 'filter_query': '{turnover_rate} < 0.5'},
                             'backgroundColor': '#f8d7da', 'color': '#721c24'}
                        ]
                    ),
                    dcc.Graph(id='turnover-bar'),
                    html.H4("Stock Age Metrics", className="text-center mt-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='stock-age-table',
                        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in stock_age_df.columns],
                        data=stock_age_df.to_dict('records'),
                        filter_action='native',
                        sort_action='native',
                        page_size=10,
                        style_table={'overflowX': 'auto', 'borderRadius': '8px', 'className': 'data-table'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'stock_age_days', 'filter_query': '{stock_age_days} > 365'},
                             'backgroundColor': '#f8d7da', 'color': '#721c24'}
                        ]
                    ),
                    dcc.Graph(id='stock-age-bar'),
                    html.H4("Demand Variability Metrics", className="text-center mt-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='demand-variability-table',
                        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in demand_variability_df.columns],
                        data=demand_variability_df.to_dict('records'),
                        filter_action='native',
                        sort_action='native',
                        page_size=10,
                        style_table={'overflowX': 'auto', 'borderRadius': '8px', 'className': 'data-table'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'variability_index', 'filter_query': '{variability_index} > 0.5'},
                             'backgroundColor': '#ffdab9', 'color': '#804000'}
                        ]
                    ),
                    dcc.Graph(id='demand-variability-bar'),
                    html.H4("Stock Value Trend Metrics", className="text-center mt-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='stock-value-trend-table',
                        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in stock_value_trend_df.columns],
                        data=stock_value_trend_df.to_dict('records'),
                        filter_action='native',
                        sort_action='native',
                        page_size=10,
                        style_table={'overflowX': 'auto', 'borderRadius': '8px', 'className': 'data-table'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_data_conditional=[
                            {'if': {'column_id': 'value_per_unit', 'filter_query': '{value_per_unit} > 100'},
                             'backgroundColor': '#d4edda', 'color': '#155724'}
                        ]
                    ),
                    dcc.Graph(id='stock-value-trend-bar'),
                    dcc.Graph(id='demand-vs-variability-scatter'),
                    dcc.Loading(
                        type="circle",
                        children=dbc.Button('Export to PDF', id='export-analytics-pdf', color='info', className='mt-3')
                    ),
                    html.Div(id='analytics-output', className='text-success mt-2 text-center')
                ]), className="shadow mb-5 bg-white rounded p-4")
            ]),
            dcc.Tab(label='User Management', value='tab-users', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("User Management", className="text-center mb-4", style={'color': '#007bff'}),
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
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'}
                    ),
                    dbc.Row([
                        dbc.Col(dbc.Button('Add Row', id='add-user-row', color='primary', className='mr-2')),
                        dbc.Col(dbc.Button('Save to CSV', id='save-users', color='success', className='mr-2')),
                        dbc.Col(dcc.Loading(
                            type="circle",
                            children=dbc.Button('Export to PDF', id='export-users-pdf', color='info')
                        )),
                    ], className="my-3", justify="center"),
                    html.Div(id='users-output', className='text-success mt-2 text-center')
                ]), className="shadow mb-5 bg-white rounded p-4")
            ]),
            dcc.Tab(label='Audit Trail', value='tab-audit', children=[
                dbc.Card(dbc.CardBody([
                    html.H3("Audit Trail", className="text-center mb-4", style={'color': '#007bff'}),
                    dash_table.DataTable(
                        id='audit-table',
                        columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in audit_log.columns],
                        data=audit_log.to_dict('records'),
                        filter_action='native',
                        sort_action='native',
                        page_size=10,
                        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
                        style_cell={'padding': '10px', 'fontSize': '14px', 'color': '#333333', 'backgroundColor': '#ffffff', 'textAlign': 'center', 'border': '1px solid #dee2e6'},
                        style_header={'backgroundColor': '#e9ecef', 'color': '#495057', 'fontWeight': 'bold', 'textAlign': 'center', 'border': '1px solid #dee2e6'}
                    ),
                    dcc.Loading(
                        type="circle",
                        children=dbc.Button('Export to PDF', id='export-audit-pdf', color='info', className='mt-3')
                    ),
                    html.Div(id='audit-output', className='text-success mt-2 text-center')
                ]), className="shadow mb-5 bg-white rounded p-4")
            ])
        ])
    ], style={'background-color': '#f8f9fa'})
])

# App layout
app.layout = html.Div([
    dcc.Store(id='login-status', storage_type='session'),
    html.Div(id='page-content')
])

# Update page content based on login status
@callback(
    Output('page-content', 'children'),
    Input('login-status', 'data')
)
def render_page_content(data):
    if data and data['logged_in']:
        return dashboard_layout
    return login_layout

# Login callback
@callback(
    [Output('login-status', 'data'),
     Output('login-output', 'children')],
    Input('login-button', 'n_clicks'),
    [State('username', 'value'),
     State('password', 'value')],
    prevent_initial_call=True
)
def login(n, username, password):
    if n:
        if username and password:
            hashed_pw = hashlib.md5(password.encode()).hexdigest()
            user = df_users[df_users['username'] == username]
            if not user.empty and user['password'].iloc[0] == hashed_pw:
                role = user['role'].iloc[0]
                audit_log.loc[len(audit_log)] = [datetime.now(), username, 'Login', 'Successful login']
                audit_log.to_csv(AUDIT_PATH, index=False)
                logging.info(f"User {username} logged in successfully")
                return {'logged_in': True, 'username': username, 'role': role}, ''
            else:
                audit_log.loc[len(audit_log)] = [datetime.now(), username, 'Login Attempt', 'Failed login attempt']
                audit_log.to_csv(AUDIT_PATH, index=False)
                logging.warning(f"Failed login attempt for {username}")
                return {'logged_in': False}, 'Invalid username or password'
        return {'logged_in': False}, 'Please enter username and password'
    raise PreventUpdate

# Logout callback
@callback(
    Output('login-status', 'data', allow_duplicate=True),
    Input('logout-button', 'n_clicks'),
    State('login-status', 'data'),
    prevent_initial_call=True
)
def logout(n, data):
    if n and data['logged_in']:
        audit_log.loc[len(audit_log)] = [datetime.now(), data['username'], 'Logout', 'User logged out']
        audit_log.to_csv(AUDIT_PATH, index=False)
        logging.info(f"User {data['username']} logged out")
        return {'logged_in': False}
    raise PreventUpdate

# Update user greeting
@callback(
    Output('user-greeting', 'children'),
    Input('login-status', 'data')
)
def update_greeting(data):
    if data and data['logged_in']:
        return f"Welcome, {data['username']} ({data['role'].capitalize()})"
    return ''

# Update KPIs and Insights
@callback(
    [Output('kpi-total-value', 'children'),
     Output('kpi-expired', 'children'),
     Output('kpi-low-stock', 'children'),
     Output('kpi-consumables-cost', 'children'),
     Output('kpi-a-items', 'children'),
     Output('kpi-avg-turnover', 'children'),
     Output('kpi-delayed-shipments', 'children'),
     Output('kpi-stockout-risk', 'children'),
     Output('insights-text', 'children')],
    [Input('category-filter', 'value'),
     Input('drug-filter', 'value'),
     Input('ai-insights-button', 'n_clicks'),
     Input('interval-update', 'n_intervals')]
)
def update_kpis_and_insights(categories, drugs, n_clicks, n_intervals):
    df = df_inventory.copy()
    if categories:
        df = df[df['category'].isin(categories)]
    if drugs:
        df = df[df['drug_name'].isin(drugs)]
    total_value = df['total_value'].sum()
    expired = df['expired'].sum()
    low_stock = df['low_stock'].sum()
    consumables_cost = df_consumables['total_cost'].sum()
    a_items = df[df['abc_category'] == 'A'].shape[0]
    avg_turnover = turnover_df['turnover_rate'].mean() if not turnover_df.empty else 0
    delayed_shipments = num_delayed_shipments
    stockout_risk = df[df['stockout_risk'] == 'High Risk'].shape[0]
    top_low_stock = df[df['low_stock']].sort_values('quantity')['drug_name'].head(5).to_list()
    avg_price = df['price_per_unit'].mean()
    reorder_alerts = df[df['quantity'] < df['reorder_point']]['drug_name'].to_list()
    min_stock_alerts = df[df['quantity'] < df['minimum_stock']]['drug_name'].to_list()
    delayed_shipments_list = df_shipments[df_shipments['status'] == 'Delayed']['product'].to_list()
    abc_a = df[df['abc_category'] == 'A']['drug_name'].to_list()
    stockout_risks = df[df['stockout_risk'] == 'High Risk']['drug_name'].to_list()
    if n_clicks or n_intervals:
        insights = (f"**Filtered Total Value:** ${total_value:,.2f}\n**Expired Items:** {expired}\n**Low Stock Items:** {low_stock}\n"
                    f"**Top Low Stock Drugs:** {', '.join(top_low_stock)}\n**Average Price per Unit:** ${avg_price:.2f}\n"
                    f"**Reorder Alerts:** Reorder {', '.join(reorder_alerts)} due to 6-month lead time.\n"
                    f"**Minimum Stock Alerts:** {', '.join(min_stock_alerts)} below 6 months buffer.\n"
                    f"**Delayed Shipments:** {', '.join(delayed_shipments_list)}.\n**ABC A-Class Items:** {', '.join(abc_a)} (high priority).\n"
                    f"**Stockout Risks:** {', '.join(stockout_risks)} at high risk.\n"
                    f"**AI Recommendation:** Place orders for {', '.join(reorder_alerts)} now; consider air freight for urgent items to reduce lead time. Monitor delayed shipments and adjust forecasts.")
    else:
        insights = f"**Filtered Total Value:** ${total_value:,.2f}\n**Expired Items:** {expired}\n**Low Stock Items:** {low_stock}\n**Top Low Stock Drugs:** {', '.join(top_low_stock)}"
    return (
        f"${total_value:,.2f}",
        expired,
        low_stock,
        f"${consumables_cost:,.2f}",
        f"{a_items} ({a_value_percent:.1f}%)",
        f"{avg_turnover:.2f}",
        delayed_shipments,
        stockout_risk,
        dcc.Markdown(insights)
    )

# Update all charts
@callback(
    [Output('inventory-bar', 'figure'),
     Output('category-pie', 'figure'),
     Output('executive-supplier-map', 'figure'),
     Output('consumables-bar', 'figure'),
     Output('history-line', 'figure'),
     Output('price-quantity-scatter', 'figure'),
     Output('quantity-heatmap', 'figure'),
     Output('expiration-timeline', 'figure'),
     Output('supplier-count-bar', 'figure'),
     Output('value-treemap', 'figure'),
     Output('quantity-box', 'figure'),
     Output('consumables-cost-pie', 'figure'),
     Output('supplier-geo-map', 'figure'),
     Output('abc-pie', 'figure'),
     Output('shipment-timeline', 'figure'),
     Output('supplier-performance-bar', 'figure'),
     Output('batch-heatmap', 'figure'),
     Output('turnover-bar', 'figure'),
     Output('stock-age-bar', 'figure'),
     Output('demand-variability-bar', 'figure'),
     Output('stock-value-trend-bar', 'figure'),
     Output('reorder-vs-quantity-bar', 'figure'),
     Output('stockout-risk-pie', 'figure'),
     Output('demand-vs-variability-scatter', 'figure')],
    [Input('category-filter', 'value'),
     Input('drug-filter', 'value'),
     Input('theme-toggle', 'value'),
     Input('interval-update', 'n_intervals')]
)
def update_all_charts(categories, drugs, theme, n_intervals):
    df_inv = df_inventory.copy()
    df_hist = df_history.copy()
    if categories:
        df_inv = df_inv[df_inv['category'].isin(categories)]
        df_hist = df_hist[df_hist['drug_name'].isin(df_inv['drug_name'])]
    if drugs:
        df_inv = df_inv[df_inv['drug_name'].isin(drugs)]
        df_hist = df_hist[df_hist['drug_name'].isin(drugs)]
    
    if df_inv.empty or df_hist.empty or df_consumables.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title='No Data Available', showlegend=False)
        return [empty_fig] * 24

    inventory_bar = px.bar(
        df_inv.sort_values('quantity'),
        y='drug_name',
        x='quantity',
        color='expired',
        orientation='h',
        title='Inventory Levels (Red: Expired)',
        color_discrete_map={True: '#dc3545', False: '#28a745'},
        hover_data=['expiration_date', 'total_value', 'supplier', 'reorder_alert', 'abc_category']
    )
    
    category_pie = px.pie(
        df_inv,
        values='total_value',
        names='category',
        title='Inventory Value by Category',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    executive_supplier_map = px.scatter_geo(
        df_suppliers,
        lat='lat',
        lon='lon',
        hover_name='supplier_name',
        hover_data=['country', 'contact_email', 'location_name'],
        title='Supplier Locations (Executive View)',
        projection='natural earth',
        size=[10] * len(df_suppliers),
        color_discrete_sequence=['#007bff']
    )
    
    consumables_bar = px.bar(
        df_consumables.sort_values('quantity'),
        y='item_name',
        x='quantity',
        orientation='h',
        title='Consumables Inventory Levels',
        color_discrete_sequence=['#28a745'],
        hover_data=['category', 'total_cost', 'supplier', 'reorder_alert']
    )
    
    history_line = go.Figure()
    for drug in df_hist['drug_name'].unique():
        df_drug = df_hist[df_hist['drug_name'] == drug]
        if len(df_drug) >= 2:
            history_line.add_trace(go.Scatter(
                x=df_drug['date'],
                y=df_drug['quantity'],
                name=drug,
                mode='lines+markers',
                line=dict(width=2)
            ))
            future_dates, predicted, ci = forecast_stock(df_drug['date'], df_drug['quantity'])
            if not future_dates.empty:
                history_line.add_trace(go.Scatter(
                    x=future_dates,
                    y=predicted,
                    name=f'{drug} Forecast',
                    mode='lines',
                    line=dict(dash='dash', width=1)
                ))
                history_line.add_trace(go.Scatter(
                    x=future_dates,
                    y=predicted + ci,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                history_line.add_trace(go.Scatter(
                    x=future_dates,
                    y=predicted - ci,
                    fill='tonexty',
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    opacity=0.2
                ))
    history_line.update_layout(
        title='Stock History with 90-Day Forecast (95% CI)',
        xaxis_title='Date',
        yaxis_title='Quantity',
        hovermode='x unified'
    )
    
    price_quantity_scatter = px.scatter(
        df_inv,
        x='quantity',
        y='price_per_unit',
        color='category',
        size='total_value',
        hover_data=['drug_name', 'supplier', 'reorder_alert', 'abc_category'],
        title='Price per Unit vs Quantity (Size by Value)'
    )
    
    heatmap_data = df_inv.groupby(['category', 'supplier'])['quantity'].sum().unstack().fillna(0)
    quantity_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Supplier", y="Category", color="Quantity"),
        title='Quantity Heatmap by Category and Supplier'
    )
    
    df_inv_with_start['x_end'] = df_inv_with_start['expiration_date']
    expiration_timeline = px.timeline(
        df_inv_with_start,
        x_start='x_start',
        x_end='x_end',
        y='drug_name',
        color='quantity',
        title='Expiration Timeline for Drugs'
    )
    expiration_timeline.update_layout(xaxis_title='Date', yaxis_title='Drug Name')
    
    supplier_count_df = df_inv['supplier'].value_counts().reset_index()
    supplier_count_df.columns = ['supplier', 'count']
    supplier_count_bar = px.bar(
        supplier_count_df,
        x='supplier',
        y='count',
        title='Number of Drugs per Supplier'
    )
    
    value_treemap = px.treemap(
        df_inv,
        path=['category', 'drug_name'],
        values='total_value',
        title='Inventory Value Treemap by Category and Drug'
    )
    
    quantity_box = px.box(
        df_inv,
        x='category',
        y='quantity',
        color='category',
        title='Quantity Distribution by Category'
    )
    
    consumables_cost_pie = px.pie(
        df_consumables,
        values='total_cost',
        names='category',
        title='Consumables Cost by Category'
    )
    
    supplier_geo_map = px.scatter_geo(
        df_suppliers,
        lat='lat',
        lon='lon',
        hover_name='supplier_name',
        hover_data=['country', 'contact_email', 'location_name'],
        title='Supplier Locations Across India, Europe, and Africa',
        projection='natural earth',
        size=[10] * len(df_suppliers),
        color_discrete_sequence=['#007bff']
    )
    
    abc_pie = px.pie(
        df_inv,
        values='total_value',
        names='abc_category',
        title='ABC Analysis by Value',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    df_shipments_with_start = df_shipments.copy()
    df_shipments_with_start['start'] = df_shipments_with_start['order_date']
    df_shipments_with_start['end'] = df_shipments_with_start['expected_arrival']
    shipment_timeline = px.timeline(
        df_shipments_with_start,
        x_start='start',
        x_end='end',
        y='product',
        color='status',
        title='Shipment Timeline'
    )
    shipment_timeline.update_layout(xaxis_title='Date', yaxis_title='Product')
    
    supplier_performance_bar = px.bar(
        supplier_performance,
        x='supplier',
        y='delay_percentage',
        title='Supplier Reliability (Delay Percentage)',
        color='delay_percentage',
        color_continuous_scale='Reds',
        hover_data=['num_shipments', 'on_time_delivery_rate']
    )
    
    batch_heatmap_data = batch_data.groupby(['drug_name', 'status'])['quantity'].sum().unstack().fillna(0)
    batch_heatmap = px.imshow(
        batch_heatmap_data,
        labels=dict(x="Status", y="Drug Name", color="Quantity"),
        title='Batch Status Heatmap'
    )
    
    turnover_bar = px.bar(
        turnover_df.sort_values('turnover_rate'),
        x='drug_name',
        y='turnover_rate',
        title='Inventory Turnover Rate by Drug',
        color='turnover_rate',
        color_continuous_scale='Viridis',
        hover_data=['total_sold', 'average_stock']
    )
    
    stock_age_bar = px.bar(
        stock_age_df.sort_values('stock_age_days'),
        x='drug_name',
        y='stock_age_days',
        title='Stock Age by Drug (Days)',
        color='stock_age_days',
        color_continuous_scale='Reds',
        hover_data=['quantity']
    )
    
    demand_variability_bar = px.bar(
        demand_variability_df.sort_values('variability_index'),
        x='drug_name',
        y='variability_index',
        title='Demand Variability Index by Drug',
        color='variability_index',
        color_continuous_scale='Blues',
        hover_data=['average_daily_demand', 'std_dev_demand']
    )
    
    stock_value_trend_bar = px.bar(
        stock_value_trend_df.sort_values('value_per_unit'),
        x='drug_name',
        y='value_per_unit',
        title='Stock Value per Unit by Drug',
        color='value_per_unit',
        color_continuous_scale='Greens',
        hover_data=['total_value', 'quantity']
    )
    
    reorder_vs_quantity_bar = px.bar(
        df_inv,
        x='drug_name',
        y=['quantity', 'reorder_point'],
        barmode='group',
        title='Reorder Point vs Current Quantity'
    )
    
    stockout_risk_pie = px.pie(
        df_inv,
        values='total_value',
        names='stockout_risk',
        title='Stockout Risk by Value',
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    demand_vs_variability_scatter = px.scatter(
        demand_variability_df,
        x='average_daily_demand',
        y='variability_index',
        color='drug_name',
        title='Demand vs Variability Scatter'
    )
    
    plot_bg = '#2c3e50' if theme == 'Dark' else '#f8f9fa'
    paper_bg = '#2c3e50' if theme == 'Dark' else '#f8f9fa'
    font_color = '#ffffff' if theme == 'Dark' else '#3498db'
    for fig in [inventory_bar, category_pie, executive_supplier_map, consumables_bar, history_line, price_quantity_scatter, 
                quantity_heatmap, expiration_timeline, supplier_count_bar, value_treemap, quantity_box, consumables_cost_pie, 
                supplier_geo_map, abc_pie, shipment_timeline, supplier_performance_bar, batch_heatmap, turnover_bar, stock_age_bar, 
                demand_variability_bar, stock_value_trend_bar, reorder_vs_quantity_bar, stockout_risk_pie, demand_vs_variability_scatter]:
        fig.update_layout(
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            font_color=font_color,
            title_font_size=18,
            showlegend=True,
            margin=dict(l=20, r=20, t=50, b=20),
            transition_duration=500
        )
    return (inventory_bar, category_pie, executive_supplier_map, consumables_bar, history_line, 
            price_quantity_scatter, quantity_heatmap, expiration_timeline, supplier_count_bar, 
            value_treemap, quantity_box, consumables_cost_pie, supplier_geo_map, abc_pie, 
            shipment_timeline, supplier_performance_bar, batch_heatmap, turnover_bar, stock_age_bar, 
            demand_variability_bar, stock_value_trend_bar, reorder_vs_quantity_bar, stockout_risk_pie, 
            demand_vs_variability_scatter)

# Forecasting graph and table callback
@callback(
    [Output('forecast-graph', 'figure'),
     Output('forecast-table', 'data')],
    Input('forecast-drug', 'value'),
    Input('theme-toggle', 'value')
)
def update_forecast_graph(drug, theme):
    if not drug:
        return go.Figure(layout={'title': 'Select a drug to view forecast'}), []
    df_drug = df_history[df_history['drug_name'] == drug]
    if len(df_drug) < 2:
        return go.Figure(layout={'title': 'Insufficient data for forecast'}), []
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_drug['date'], y=df_drug['quantity'], name=drug, mode='lines+markers'))
    future_dates, predicted, ci = forecast_stock(df_drug['date'], df_drug['quantity'])
    forecast_data = []
    if not future_dates.empty:
        fig.add_trace(go.Scatter(x=future_dates, y=predicted, name='Forecast', mode='lines', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=future_dates, y=predicted + ci, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=future_dates, y=predicted - ci, fill='tonexty', mode='lines', line=dict(width=0), showlegend=False, opacity=0.2))
        for date, pred in zip(future_dates, predicted):
            forecast_data.append({'date': date.strftime('%Y-%m-%d'), 'predicted': round(pred, 2)})
    fig.update_layout(title=f'{drug} Stock Forecast (90 Days)', xaxis_title='Date', yaxis_title='Quantity')
    plot_bg = '#2c3e50' if theme == 'Dark' else '#f8f9fa'
    paper_bg = '#2c3e50' if theme == 'Dark' else '#f8f9fa'
    font_color = '#ffffff' if theme == 'Dark' else '#3498db'
    fig.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg, font_color=font_color)
    return fig, forecast_data

# Add row to inventory
@callback(
    [Output('inventory-table', 'data'),
     Output('inv-output', 'children')],
    Input('add-inv-row', 'n_clicks'),
    [State('inventory-table', 'data'),
     State('inventory-table', 'columns'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def add_inv_row(n, rows, cols, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to add inventory row")
        return rows, 'Please log in to perform this action'
    if n and login_status['role'] in ['admin', 'manager']:
        try:
            new_row = {c['id']: '' for c in cols}
            new_row['id'] = max([int(row['id']) for row in rows], default=0) + 1
            new_row['lead_time_days'] = 180
            new_row['average_daily_demand'] = 0
            new_row['std_dev_demand'] = 0
            rows.append(new_row)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Add Row', 'Added new row to inventory']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} added new inventory row")
            return rows, 'New inventory row added successfully'
        except Exception as e:
            logging.error(f"Error adding inventory row: {e}")
            return rows, f'Error adding row: {e}'
    return rows, 'Permission denied: Only admins and managers can add rows'

# Save inventory
@callback(
    Output('inv-output', 'children', allow_duplicate=True),
    Input('save-inv', 'n_clicks'),
    [State('inventory-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def save_inv(n, data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to save inventory")
        return 'Please log in to perform this action'
    if n and login_status['role'] in ['admin', 'manager']:
        try:
            df = pd.DataFrame(data)
            if df['quantity'].apply(lambda x: pd.isna(x) or not isinstance(x, (int, float)) or x < 0).any():
                logging.error("Invalid quantity in inventory data")
                return 'Error: Quantity must be non-negative numbers'
            if df['price_per_unit'].apply(lambda x: pd.isna(x) or not isinstance(x, (int, float)) or x < 0).any():
                logging.error("Invalid price in inventory data")
                return 'Error: Price must be non-negative numbers'
            if df['expiration_date'].isna().all():
                logging.error("Invalid expiration date in inventory data")
                return 'Error: All rows must have valid expiration dates'
            df.to_csv(INVENTORY_PATH, index=False)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Save Inventory', 'Saved inventory changes']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} saved inventory changes")
            return 'Inventory saved successfully'
        except Exception as e:
            logging.error(f"Error saving inventory: {e}")
            return f'Error saving inventory: {e}'
    return 'Permission denied: Only admins and managers can save inventory'

# Export inventory to PDF
@callback(
    Output('inv-output', 'children', allow_duplicate=True),
    Input('export-inv-pdf', 'n_clicks'),
    [State('inventory-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def export_inv_pdf(n, data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to export inventory PDF")
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            col_width = pdf.w / len(data[0]) if data else pdf.w / 12
            for key in data[0].keys() if data else []:
                pdf.cell(col_width, 10, str(key), border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in data or [{}]:
                for value in row.values():
                    pdf.cell(col_width, 10, str(value)[:20], border=1)
                pdf.ln()
            pdf_file = os.path.join(REPORTS_PATH, f'inventory_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export PDF', 'Exported inventory to PDF']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported inventory to PDF")
            return f'Inventory PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting inventory PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

# Generate reorder suggestions
@callback(
    [Output('reorder-table', 'data'),
     Output('reorder-output', 'children')],
    Input('reorder-suggestions', 'n_clicks'),
    [State('inventory-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def generate_reorder(n, data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to generate reorder suggestions")
        return [], 'Please log in to perform this action'
    if n and login_status['role'] in ['admin', 'manager']:
        try:
            df = pd.DataFrame(data)
            suggestions = generate_reorder_suggestions(df)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Generate Reorder Suggestions', 'Generated reorder suggestions']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} generated reorder suggestions")
            return suggestions.to_dict('records'), 'Reorder suggestions generated successfully'
        except Exception as e:
            logging.error(f"Error generating reorder suggestions: {e}")
            return [], f'Error generating suggestions: {e}'
    return [], 'Permission denied: Only admins and managers can generate suggestions'

# Generate stockout risks
@callback(
    [Output('stockout-risk-table', 'data'),
     Output('stockout-output', 'children')],
    Input('reorder-suggestions', 'n_clicks'),
    [State('inventory-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def generate_stockout(n, data, login_status):
    if not login_status['logged_in']:
        return [], 'Please log in to perform this action'
    if n and login_status['role'] in ['admin', 'manager']:
        try:
            df = pd.DataFrame(data)
            risks = generate_stockout_risks(df)
            return risks.to_dict('records'), 'Stockout risks generated successfully'
        except Exception as e:
            return [], f'Error generating stockout risks: {e}'
    return [], 'Permission denied'

# Add row to consumables
@callback(
    [Output('consumables-table', 'data'),
     Output('cons-output', 'children')],
    Input('add-cons-row', 'n_clicks'),
    [State('consumables-table', 'data'),
     State('consumables-table', 'columns'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def add_cons_row(n, rows, cols, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to add consumables row")
        return rows, 'Please log in to perform this action'
    if n and login_status['role'] in ['admin', 'manager']:
        try:
            new_row = {c['id']: '' for c in cols}
            new_row['id'] = max([int(row['id']) for row in rows], default=0) + 1
            new_row['lead_time_days'] = 180
            new_row['average_daily_demand'] = 0
            new_row['std_dev_demand'] = 0
            rows.append(new_row)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Add Row', 'Added new row to consumables']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} added new consumables row")
            return rows, 'New consumables row added successfully'
        except Exception as e:
            logging.error(f"Error adding consumables row: {e}")
            return rows, f'Error adding row: {e}'
    return rows, 'Permission denied: Only admins and managers can add rows'

# Save consumables
@callback(
    Output('cons-output', 'children', allow_duplicate=True),
    Input('save-cons', 'n_clicks'),
    [State('consumables-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def save_cons(n, data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to save consumables")
        return 'Please log in to perform this action'
    if n and login_status['role'] in ['admin', 'manager']:
        try:
            df = pd.DataFrame(data)
            if df['quantity'].apply(lambda x: pd.isna(x) or not isinstance(x, (int, float)) or x < 0).any():
                logging.error("Invalid quantity in consumables data")
                return 'Error: Quantity must be non-negative numbers'
            if df['price_per_unit'].apply(lambda x: pd.isna(x) or not isinstance(x, (int, float)) or x < 0).any():
                logging.error("Invalid price in consumables data")
                return 'Error: Price must be non-negative numbers'
            df.to_csv(CONSUMABLES_PATH, index=False)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Save Consumables', 'Saved consumables changes']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} saved consumables changes")
            return 'Consumables saved successfully'
        except Exception as e:
            logging.error(f"Error saving consumables: {e}")
            return f'Error saving consumables: {e}'
    return 'Permission denied: Only admins and managers can save consumables'

# Export consumables to PDF
@callback(
    Output('cons-output', 'children', allow_duplicate=True),
    Input('export-cons-pdf', 'n_clicks'),
    [State('consumables-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def export_cons_pdf(n, data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to export consumables PDF")
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            col_width = pdf.w / len(data[0]) if data else pdf.w / 11
            for key in data[0].keys() if data else []:
                pdf.cell(col_width, 10, str(key), border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in data or [{}]:
                for value in row.values():
                    pdf.cell(col_width, 10, str(value)[:20], border=1)
                pdf.ln()
            pdf_file = os.path.join(REPORTS_PATH, f'consumables_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export PDF', 'Exported consumables to PDF']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported consumables to PDF")
            return f'Consumables PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting consumables PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

# Add row to shipment
@callback(
    [Output('shipment-table', 'data'),
     Output('shipment-output', 'children')],
    Input('add-shipment-row', 'n_clicks'),
    [State('shipment-table', 'data'),
     State('shipment-table', 'columns'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def add_shipment_row(n, rows, cols, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to add shipment row")
        return rows, 'Please log in to perform this action'
    if n and login_status['role'] in ['admin', 'manager']:
        try:
            new_row = {c['id']: '' for c in cols}
            new_row['id'] = max([int(row['id']) for row in rows], default=0) + 1
            new_row['order_date'] = current_date.strftime('%Y-%m-%d')
            new_row['expected_arrival'] = (current_date + timedelta(days=180)).strftime('%Y-%m-%d')
            new_row['status'] = 'Ordered'
            rows.append(new_row)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Add Shipment', 'Added new shipment row']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} added new shipment row")
            return rows, 'New shipment added successfully'
        except Exception as e:
            logging.error(f"Error adding shipment row: {e}")
            return rows, f'Error adding shipment: {e}'
    return rows, 'Permission denied: Only admins and managers can add shipments'

# Save shipments
@callback(
    Output('shipment-output', 'children', allow_duplicate=True),
    Input('save-shipments', 'n_clicks'),
    [State('shipment-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def save_shipments(n, data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to save shipments")
        return 'Please log in to perform this action'
    if n and login_status['role'] in ['admin', 'manager']:
        try:
            df = pd.DataFrame(data)
            df.to_csv(SHIPMENTS_PATH, index=False)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Save Shipments', 'Saved shipment changes']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} saved shipment changes")
            return 'Shipments saved successfully'
        except Exception as e:
            logging.error(f"Error saving shipments: {e}")
            return f'Error saving shipments: {e}'
    return 'Permission denied: Only admins and managers can save shipments'

# Export shipments to PDF
@callback(
    Output('shipment-output', 'children', allow_duplicate=True),
    Input('export-shipments-pdf', 'n_clicks'),
    [State('shipment-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def export_shipments_pdf(n, data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to export shipments PDF")
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            col_width = pdf.w / len(data[0]) if data else pdf.w / 7
            for key in data[0].keys() if data else []:
                pdf.cell(col_width, 10, str(key), border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in data or [{}]:
                for value in row.values():
                    pdf.cell(col_width, 10, str(value)[:20], border=1)
                pdf.ln()
            pdf_file = os.path.join(REPORTS_PATH, f'shipments_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export PDF', 'Exported shipments to PDF']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported shipments to PDF")
            return f'Shipments PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting shipments PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

# Export inventory metrics to PDF
@callback(
    Output('metrics-output', 'children', allow_duplicate=True),
    Input('export-metrics-pdf', 'n_clicks'),
    [State('inventory-metrics-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def export_metrics_pdf(n, data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to export metrics PDF")
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            col_width = pdf.w / len(data[0]) if data else pdf.w / 7
            for key in data[0].keys() if data else []:
                pdf.cell(col_width, 10, str(key), border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in data or [{}]:
                for value in row.values():
                    pdf.cell(col_width, 10, str(value)[:20], border=1)
                pdf.ln()
            pdf_file = os.path.join(REPORTS_PATH, f'metrics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export PDF', 'Exported metrics to PDF']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported metrics to PDF")
            return f'Metrics PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting metrics PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

# Export suppliers to PDF
@callback(
    Output('suppliers-output', 'children', allow_duplicate=True),
    Input('export-suppliers-pdf', 'n_clicks'),
    [State('suppliers-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def export_suppliers_pdf(n, data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to export suppliers PDF")
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            col_width = pdf.w / len(data[0]) if data else pdf.w / 3
            for key in data[0].keys() if data else []:
                pdf.cell(col_width, 10, str(key), border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in data or [{}]:
                for value in row.values():
                    pdf.cell(col_width, 10, str(value)[:20], border=1)
                pdf.ln()
            pdf_file = os.path.join(REPORTS_PATH, f'suppliers_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export PDF', 'Exported suppliers to PDF']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported suppliers to PDF")
            return f'Suppliers PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting suppliers PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

# Export batch tracking to PDF
@callback(
    Output('batch-output', 'children', allow_duplicate=True),
    Input('export-batch-pdf', 'n_clicks'),
    [State('batch-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def export_batch_pdf(n, data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to export batch PDF")
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            col_width = pdf.w / len(data[0]) if data else pdf.w / 5
            for key in data[0].keys() if data else []:
                pdf.cell(col_width, 10, str(key), border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in data or [{}]:
                for value in row.values():
                    pdf.cell(col_width, 10, str(value)[:20], border=1)
                pdf.ln()
            pdf_file = os.path.join(REPORTS_PATH, f'batch_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export PDF', 'Exported batch tracking to PDF']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported batch tracking to PDF")
            return f'Batch Tracking PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting batch PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

# Export analytics to PDF
@callback(
    Output('analytics-output', 'children', allow_duplicate=True),
    Input('export-analytics-pdf', 'n_clicks'),
    [State('analytics-table', 'data'),
     State('stock-age-table', 'data'),
     State('demand-variability-table', 'data'),
     State('stock-value-trend-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def export_analytics_pdf(n, turnover_data, stock_age_data, demand_var_data, stock_value_data, login_status):
    if not login_status['logged_in']:
        logging.warning("Unauthorized attempt to export analytics PDF")
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = PDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Inventory Analytics Report', 0, 1, 'C')
            pdf.ln(10)

            # Turnover Section
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Turnover Metrics', 0, 1)
            col_width = pdf.w / 4
            for key in ['drug_name', 'total_sold', 'average_stock', 'turnover_rate']:
                pdf.cell(col_width, 10, key, border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in turnover_data:
                for value in [row['drug_name'], row['total_sold'], row['average_stock'], row['turnover_rate']]:
                    pdf.cell(col_width, 10, str(value), border=1)
                pdf.ln()

            # Stock Age
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Stock Age Metrics', 0, 1)
            col_width = pdf.w / 3
            for key in stock_age_data[0].keys():
                pdf.cell(col_width, 10, key, border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in stock_age_data:
                for value in row.values():
                    pdf.cell(col_width, 10, str(value), border=1)
                pdf.ln()

            # Demand Variability
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Demand Variability Metrics', 0, 1)
            col_width = pdf.w / 4
            for key in demand_var_data[0].keys():
                pdf.cell(col_width, 10, key, border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in demand_var_data:
                for value in row.values():
                    pdf.cell(col_width, 10, str(value), border=1)
                pdf.ln()

            # Stock Value Trend
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Stock Value Trend Metrics', 0, 1)
            col_width = pdf.w / 4
            for key in stock_value_data[0].keys():
                pdf.cell(col_width, 10, key, border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in stock_value_data:
                for value in row.values():
                    pdf.cell(col_width, 10, str(value), border=1)
                pdf.ln()

            pdf_file = os.path.join(REPORTS_PATH, f'analytics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export PDF', 'Exported analytics to PDF']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported analytics to PDF")
            return f'Analytics PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting analytics PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

# Add row to users
@callback(
    [Output('users-table', 'data'),
     Output('users-output', 'children')],
    Input('add-user-row', 'n_clicks'),
    [State('users-table', 'data'),
     State('users-table', 'columns'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def add_user_row(n, rows, cols, login_status):
    if not login_status['logged_in'] or login_status['role'] != 'admin':
        return rows, 'Permission denied: Only admins can add users'
    if n:
        try:
            new_row = {c['id']: '' for c in cols}
            new_row['id'] = max([int(row['id']) for row in rows if 'id' in row], default=0) + 1
            new_row['password'] = ''
            rows.append(new_row)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Add User', 'Added new user row']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} added new user row")
            return rows, 'New user row added'
        except Exception as e:
            logging.error(f"Error adding user row: {e}")
            return rows, f'Error: {e}'
    return rows, ''

# Save users
@callback(
    Output('users-output', 'children', allow_duplicate=True),
    Input('save-users', 'n_clicks'),
    State('users-table', 'data'),
    State('login-status', 'data'),
    prevent_initial_call=True
)
def save_users(n, data, login_status):
    if not login_status['logged_in'] or login_status['role'] != 'admin':
        return 'Permission denied: Only admins can save users'
    if n:
        try:
            df = pd.DataFrame(data)
            for idx, row in df.iterrows():
                if row.get('password', ''):  # Check if password field exists and is not empty
                    df.at[idx, 'password'] = hashlib.md5(row['password'].encode()).hexdigest()
                elif 'password' not in row:  # Ensure password column exists
                    df.at[idx, 'password'] = df_users.loc[df_users['id'] == row['id'], 'password'].iloc[0] if row['id'] in df_users['id'].values else ''
            df.to_csv(USERS_PATH, index=False)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Save Users', 'Saved user changes']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} saved user changes")
            return 'Users saved successfully'
        except Exception as e:
            logging.error(f"Error saving users: {e}")
            return f'Error: {e}'
    return ''

# Export users to PDF
@callback(
    Output('users-output', 'children', allow_duplicate=True),
    Input('export-users-pdf', 'n_clicks'),
    State('users-table', 'data'),
    State('login-status', 'data'),
    prevent_initial_call=True
)
def export_users_pdf(n, data, login_status):
    if not login_status.get('logged_in', False) or login_status.get('role') != 'admin':
        return 'Permission denied'
    if n:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            col_width = pdf.w / len(data[0]) if data else pdf.w / 4
            for key in data[0].keys() if data else []:
                if key != 'password':
                    pdf.cell(col_width, 10, str(key), border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in data or [{}]:
                for key, value in row.items():
                    if key != 'password':
                        pdf.cell(col_width, 10, str(value)[:20], border=1)
                pdf.ln()
            pdf_file = os.path.join(REPORTS_PATH, f'users_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export PDF', 'Exported users to PDF']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported users to PDF")
            return f'Users PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting users PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

# Export audit trail to PDF
@callback(
    Output('audit-output', 'children', allow_duplicate=True),
    Input('export-audit-pdf', 'n_clicks'),
    [State('audit-table', 'data'),
     State('login-status', 'data')],
    prevent_initial_call=True
)
def export_audit_pdf(n, data, login_status):
    if not login_status.get('logged_in', False):
        logging.warning("Unauthorized attempt to export audit PDF")
        return 'Please log in to perform this action'
    if n:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 10)
            col_width = pdf.w / len(data[0]) if data else pdf.w / 4
            for key in data[0].keys() if data else []:
                pdf.cell(col_width, 10, str(key), border=1)
            pdf.ln()
            pdf.set_font('Arial', '', 10)
            for row in data or [{}]:
                for value in row.values():
                    pdf.cell(col_width, 10, str(value)[:20], border=1)
                pdf.ln()
            pdf_file = os.path.join(REPORTS_PATH, f'audit_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export PDF', 'Exported audit trail to PDF']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported audit trail to PDF")
            return f'Audit Trail PDF exported successfully to {pdf_file}'
        except Exception as e:
            logging.error(f"Error exporting audit PDF: {e}")
            return f'Error exporting PDF: {e}'
    return ''

# Export full report to PDF
@callback(
    [Output('alerts', 'children', allow_duplicate=True),
     Output('alerts', 'is_open', allow_duplicate=True)],
    Input('export-report', 'n_clicks'),
    State('login-status', 'data'),
    prevent_initial_call=True
)
def export_full_report(n, login_status):
    if not login_status.get('logged_in', False):
        logging.warning("Unauthorized attempt to export full report")
        return 'Please log in to perform this action', True
    if n and login_status.get('role') in ['admin']:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Unique Pharmaceuticals Full Report', 0, 1, 'C')
            pdf.ln(10)

            # Inventory Section
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Inventory Overview', 0, 1)
            pdf.ln(5)
            pdf.set_font('Arial', '', 10)
            for col in ['drug_name', 'quantity', 'total_value', 'reorder_alert', 'stockout_risk']:
                pdf.cell(40, 10, str(col), border=1)
            pdf.ln()
            for index, row in df_inventory.iterrows():
                pdf.cell(40, 10, str(row['drug_name'])[:20], border=1)
                pdf.cell(40, 10, str(row['quantity']), border=1)
                pdf.cell(40, 10, f"${row['total_value']:.2f}", border=1)
                pdf.cell(40, 10, str(row['reorder_alert']), border=1)
                pdf.cell(40, 10, str(row['stockout_risk']), border=1)
                pdf.ln()
            pdf.ln(10)

            # Consumables Section
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Consumables Overview', 0, 1)
            pdf.ln(5)
            pdf.set_font('Arial', '', 10)
            for col in ['item_name', 'quantity', 'total_cost', 'reorder_alert']:
                pdf.cell(40, 10, str(col), border=1)
            pdf.ln()
            for index, row in df_consumables.iterrows():
                pdf.cell(40, 10, str(row['item_name'])[:20], border=1)
                pdf.cell(40, 10, str(row['quantity']), border=1)
                pdf.cell(40, 10, f"${row['total_cost']:.2f}", border=1)
                pdf.cell(40, 10, str(row['reorder_alert']), border=1)
                pdf.ln()
            pdf.ln(10)

            # Shipments Section
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Shipment Tracking', 0, 1)
            pdf.ln(5)
            pdf.set_font('Arial', '', 10)
            for col in ['product', 'quantity', 'supplier', 'expected_arrival', 'status']:
                pdf.cell(40, 10, str(col), border=1)
            pdf.ln()
            for index, row in df_shipments.iterrows():
                pdf.cell(40, 10, str(row['product'])[:20], border=1)
                pdf.cell(40, 10, str(row['quantity']), border=1)
                pdf.cell(40, 10, str(row['supplier'])[:20], border=1)
                pdf.cell(40, 10, str(row['expected_arrival']), border=1)
                pdf.cell(40, 10, str(row['status']), border=1)
                pdf.ln()
            pdf.ln(10)

            # Batch Tracking Section
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 10, 'Batch Tracking', 0, 1)
            pdf.ln(5)
            pdf.set_font('Arial', '', 10)
            for col in ['batch_number', 'drug_name', 'quantity', 'expiration_date', 'status']:
                pdf.cell(40, 10, str(col), border=1)
            pdf.ln()
            for index, row in batch_data.iterrows():
                pdf.cell(40, 10, str(row.get('batch_number', ''))[:20], border=1)
                pdf.cell(40, 10, str(row.get('drug_name', ''))[:20], border=1)
                pdf.cell(40, 10, str(row.get('quantity', '')), border=1)
                pdf.cell(40, 10, str(row.get('expiration_date', '')), border=1)
                pdf.cell(40, 10, str(row.get('status', '')), border=1)
                pdf.ln()
            pdf.ln(10)

            # Save the PDF
            pdf_file = os.path.join(REPORTS_PATH, f'full_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf')
            pdf.output(pdf_file)
            audit_log.loc[len(audit_log)] = [datetime.now(), login_status['username'], 'Export Full Report', f'Exported full report to {pdf_file}']
            audit_log.to_csv(AUDIT_PATH, index=False)
            logging.info(f"User {login_status['username']} exported full report to {pdf_file}")
            return f'Full report exported successfully to {pdf_file}', True
        except Exception as e:
            logging.error(f"Error exporting full report: {e}")
            return f'Error exporting full report: {e}', True
    return "", False

# Update inventory metrics table
@callback(
    Output('inventory-metrics-table', 'data'),
    Input('interval-update', 'n_intervals')
)
def update_inventory_metrics(n_intervals):
    # Recalculate metrics if needed
    return df_inventory.to_dict('records')

if __name__ == '__main__':
    app.run(debug=False)
