import pandas as pd
from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
from dash.exceptions import PreventUpdate

# Load and preprocess data
df = pd.read_csv('E:/Codes/CLS LMS Training/Submissions/KaggleV2-May-2016.csv')

# Data cleaning
df.columns = df.columns.str.lower().str.replace('-', '_')
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
df['appointmentday'] = pd.to_datetime(df['appointmentday'])
df['no_show_numeric'] = df['no_show'].map({'Yes': 1, 'No': 0})
df['appointment_month'] = df['appointmentday'].dt.to_period('M').astype(str)
df['waiting_days'] = (df['appointmentday'] - df['scheduledday']).dt.days
df['day_of_week'] = df['appointmentday'].dt.day_name()

# Create age groups
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 18, 30, 45, 60, 100], 
                         labels=['0-18', '19-30', '31-45', '46-60', '60+'])

# Color scheme
colors = {
    'background': '#f8f9fa',
    'text': '#343a40',
    'card_background': '#ffffff',
    'primary': '#4e73df',
    'success': '#1cc88a',
    'danger': '#e74a3b',
    'warning': '#f6c23e',
    'info': '#36b9cc'
}

# Initialize the app
app = JupyterDash(__name__)

app.layout = html.Div(children=[
    # Header
    html.Div([
        html.H1(children='Medical Appointment Dashboard', 
                style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '10px'}),
        html.P('Comprehensive Analysis of Patient Appointments and No-Shows', 
               style={'textAlign': 'center', 'color': colors['text'], 'marginBottom': '30px'}),
    ], style={'backgroundColor': colors['card_background'], 'padding': '20px', 'borderRadius': '5px', 
              'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'marginBottom': '20px'}),
    
    # Filters Row
    html.Div([
        html.Div([
            html.Label('Select Gender:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Checklist(
                id='gender-checklist',
                options=[{'label': ' Male', 'value': 'M'}, {'label': ' Female', 'value': 'F'}],
                value=['M', 'F'],
                inline=True,
                inputStyle={'marginRight': '5px', 'marginLeft': '10px'}
            ),
        ], style={'flex': '1', 'marginRight': '20px'}),
        
        html.Div([
    html.Label('Select Age Groups:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
    dcc.Dropdown(
        id='age-group-dropdown',
        options=[{'label': age_group, 'value': age_group} 
                for age_group in sorted(df['age_group'].dropna().unique())],
        value=sorted(df['age_group'].dropna().unique().tolist()),
        multi=True,
        clearable=False
    ),
], style={'flex': '1', 'marginRight': '20px'}),
        
        html.Div([
            html.Label('Select Months:', style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.Dropdown(
                id='month-dropdown',
                options=[{'label': month, 'value': month} for month in sorted(df['appointment_month'].unique())],
                value=sorted(df['appointment_month'].unique()),
                multi=True,
                clearable=False
            ),
        ], style={'flex': '1'}),
    ], style={'display': 'flex', 'marginBottom': '20px', 'backgroundColor': colors['card_background'], 
              'padding': '15px', 'borderRadius': '5px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)'}),
    
    # Key Metrics Cards
    html.Div([
        html.Div(
            children=[
                html.H2(id='total-patients', style={'textAlign': 'center', 'color': colors['primary']}),
                html.P('Total Unique Patients', style={'textAlign': 'center', 'color': colors['text']}),
            ],
            style={'backgroundColor': colors['card_background'], 'padding': '20px', 'borderRadius': '5px', 
                   'margin': '10px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'flex': '1'}
        ),
        html.Div(
            children=[
                html.H2(id='total-appointments', style={'textAlign': 'center', 'color': colors['info']}),
                html.P('Total Appointments', style={'textAlign': 'center', 'color': colors['text']}),
            ],
            style={'backgroundColor': colors['card_background'], 'padding': '20px', 'borderRadius': '5px', 
                   'margin': '10px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'flex': '1'}
        ),
        html.Div(
            children=[
                html.H2(id='total-no-shows', style={'textAlign': 'center', 'color': colors['danger']}),
                html.P('Total No-Shows', style={'textAlign': 'center', 'color': colors['text']}),
            ],
            style={'backgroundColor': colors['card_background'], 'padding': '20px', 'borderRadius': '5px', 
                   'margin': '10px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'flex': '1'}
        ),
        html.Div(
            children=[
                html.H2(id='no-show-rate', style={'textAlign': 'center', 'color': colors['warning']}),
                html.P('No-Show Rate', style={'textAlign': 'center', 'color': colors['text']}),
            ],
            style={'backgroundColor': colors['card_background'], 'padding': '20px', 'borderRadius': '5px', 
                   'margin': '10px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'flex': '1'}
        ),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
    
    # First Row of Charts
    html.Div([
        html.Div(
            dcc.Graph(id='appointments-by-month-bar-chart'),
            style={'flex': '1', 'backgroundColor': colors['card_background'], 'padding': '15px', 
                   'borderRadius': '5px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'marginRight': '10px'}
        ),
        html.Div(
            dcc.Graph(id='no-show-pie-chart'),
            style={'flex': '1', 'backgroundColor': colors['card_background'], 'padding': '15px', 
                   'borderRadius': '5px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'marginLeft': '10px'}
        ),
    ], style={'display': 'flex', 'marginBottom': '20px'}),
    
    # Second Row of Charts
    html.Div([
        html.Div(
            dcc.Graph(id='age-distribution-histogram'),
            style={'flex': '1', 'backgroundColor': colors['card_background'], 'padding': '15px', 
                   'borderRadius': '5px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'marginRight': '10px'}
        ),
        html.Div(
            dcc.Graph(id='day-of-week-heatmap'),
            style={'flex': '1', 'backgroundColor': colors['card_background'], 'padding': '15px', 
                   'borderRadius': '5px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'marginLeft': '10px'}
        ),
    ], style={'display': 'flex', 'marginBottom': '20px'}),
    
    # Third Row of Charts
    html.Div([
        html.Div(
            dcc.Graph(id='waiting-days-distribution'),
            style={'flex': '1', 'backgroundColor': colors['card_background'], 'padding': '15px', 
                   'borderRadius': '5px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'marginRight': '10px'}
        ),
        html.Div(
            dcc.Graph(id='scholarship-no-show-bar'),
            style={'flex': '1', 'backgroundColor': colors['card_background'], 'padding': '15px', 
                   'borderRadius': '5px', 'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)', 'marginLeft': '10px'}
        ),
    ], style={'display': 'flex', 'marginBottom': '20px'}),
    
    # Data Table
    html.Div([
        html.H3('Sample Data', style={'color': colors['text'], 'marginBottom': '10px'}),
        dash_table.DataTable(
            id='data-table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.sample(10).to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': colors['primary'],
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={
                'backgroundColor': colors['card_background'],
                'color': colors['text'],
                'border': '1px solid #e0e0e0'
            },
            page_size=10
        )
    ], style={'backgroundColor': colors['card_background'], 'padding': '20px', 'borderRadius': '5px', 
              'boxShadow': '0 4px 6px 0 rgba(0, 0, 0, 0.1)'})
], style={'backgroundColor': colors['background'], 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

@app.callback(
    [Output('total-patients', 'children'),
     Output('total-appointments', 'children'),
     Output('total-no-shows', 'children'),
     Output('no-show-rate', 'children'),
     Output('appointments-by-month-bar-chart', 'figure'),
     Output('age-distribution-histogram', 'figure'),
     Output('no-show-pie-chart', 'figure'),
     Output('day-of-week-heatmap', 'figure'),
     Output('waiting-days-distribution', 'figure'),
     Output('scholarship-no-show-bar', 'figure'),
     Output('data-table', 'data')],
    [Input('gender-checklist', 'value'),
     Input('age-group-dropdown', 'value'),
     Input('month-dropdown', 'value')]
)
def update_dashboard(selected_genders, selected_age_groups, selected_months):
    if not selected_genders or not selected_age_groups or not selected_months:
        raise PreventUpdate
    
    filtered_df = df[
        (df['gender'].isin(selected_genders)) & 
        (df['age_group'].isin(selected_age_groups)) & 
        (df['appointment_month'].isin(selected_months))
    ]
    
    # Calculate metrics
    total_patients = filtered_df['patientid'].nunique()
    total_appointments = filtered_df.shape[0]
    total_no_shows = filtered_df[filtered_df['no_show'] == 'Yes'].shape[0]
    no_show_rate = f"{(total_no_shows / total_appointments * 100):.1f}%" if total_appointments > 0 else "0%"
    
    # Appointments by month bar chart
    monthly_appointments = filtered_df.groupby(['appointment_month', 'no_show']).size().reset_index(name='count')
    bar_fig = px.bar(
        monthly_appointments,
        x='appointment_month',
        y='count',
        color='no_show',
        barmode='group',
        title='Appointments and No-Shows by Month',
        color_discrete_map={'Yes': colors['danger'], 'No': colors['success']},
        labels={'appointment_month': 'Month', 'count': 'Number of Appointments', 'no_show': 'No-Show'}
    )
    bar_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text']
    )
    
    # Age distribution histogram
    age_distribution = px.histogram(
        filtered_df,
        x='age',
        color='no_show',
        marginal='box',
        title='Age Distribution of Appointments',
        nbins=40,
        color_discrete_map={'Yes': colors['danger'], 'No': colors['success']},
        labels={'age': 'Age', 'count': 'Number of Appointments', 'no_show': 'No-Show'}
    )
    age_distribution.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text']
    )
    
    no_show_counts = filtered_df['no_show'].value_counts().reset_index()
    no_show_counts.columns = ['no_show', 'count']
    pie_fig = px.pie(
        no_show_counts,
        values='count',
        names='no_show',
        title='Proportion of Appointments vs. No-Shows',
        color_discrete_map={'Yes': colors['danger'], 'No': colors['success']},
        hole=0.3
    )
    pie_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text']
    )
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_heatmap = filtered_df.groupby(['day_of_week', 'no_show']).size().unstack().reindex(day_order)
    heat_fig = px.imshow(
        day_heatmap,
        labels=dict(x="No-Show", y="Day of Week", color="Count"),
        x=['No', 'Yes'],
        y=day_order,
        title='Appointments by Day of Week and No-Show Status',
        color_continuous_scale=['#1cc88a', '#e74a3b']
    )
    heat_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text']
    )
    
    waiting_fig = px.box(
        filtered_df,
        x='no_show',
        y='waiting_days',
        color='no_show',
        title='Waiting Days Distribution by No-Show Status',
        color_discrete_map={'Yes': colors['danger'], 'No': colors['success']},
        labels={'waiting_days': 'Days Between Scheduling and Appointment', 'no_show': 'No-Show'}
    )
    waiting_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text']
    )
    
    scholarship_data = filtered_df.groupby(['scholarship', 'no_show']).size().unstack().reset_index()
    scholarship_fig = px.bar(
        scholarship_data,
        x='scholarship',
        y=['No', 'Yes'],
        title='No-Shows by Scholarship Status',
        labels={'value': 'Count', 'variable': 'No-Show', 'scholarship': 'Has Scholarship'},
        color_discrete_map={'Yes': colors['danger'], 'No': colors['success']}
    )
    scholarship_fig.update_layout(
        plot_bgcolor=colors['card_background'],
        paper_bgcolor=colors['card_background'],
        font_color=colors['text'],
        barmode='group'
    )
    
    table_data = filtered_df.sample(min(10, len(filtered_df))).to_dict('records')
    
    return (total_patients, total_appointments, total_no_shows, no_show_rate, 
            bar_fig, age_distribution, pie_fig, heat_fig, waiting_fig, scholarship_fig, table_data)

if __name__ == '__main__':
    app.run(debug=True, port=8030)