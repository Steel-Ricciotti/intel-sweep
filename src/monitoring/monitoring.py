import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import logging
from typing import List, Dict, Any
import mlflow
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Monitoring:
    """Manages monitoring and visualization for ML experiments."""
    
    def __init__(self, output_dir: str):
        """
        Initialize Monitoring with output directory.
        
        Args:
            output_dir (str): Directory for saving dashboard artifacts.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.app = Dash(__name__)

    def create_dashboard(self, results: List[Dict[str, Any]]) -> None:
        """
        Create an interactive dashboard to visualize experiment results with nested tabs for tickers and models.
        
        Args:
            results (List[Dict[str, Any]]): List of dicts with ticker (e.g., 'BTC-USD_XGBoost'), horizon, metrics, and backtest_df.
        """
        logger.info("Creating dashboard for experiment results")
        if not results:
            logger.error("No results provided for dashboard")
            return

        # Prepare metrics table
        metrics_data = []
        for result in results:
            ticker_full = result['ticker']
            try:
                ticker, model = ticker_full.split('_', 1)  # Split 'BTC-USD_XGBoost' into 'BTC-USD', 'XGBoost'
            except ValueError:
                logger.warning(f"Invalid ticker format: {ticker_full}, skipping")
                continue
            metrics = result['metrics']
            metrics_data.append({
                'Ticker': ticker,
                'Model': model,
                'Horizon': result['horizon'],
                'RMSE': round(metrics['rmse'], 4),
                'MAE': round(metrics['mae'], 4),
                'R²': round(metrics['r2'], 4)
            })
        metrics_df = pd.DataFrame(metrics_data)

        # Organize results by ticker and model
        ticker_results = {}
        for result in results:
            ticker_full = result['ticker']
            try:
                ticker, model = ticker_full.split('_', 1)
                if ticker not in ticker_results:
                    ticker_results[ticker] = {}
                ticker_results[ticker][model] = result
            except ValueError:
                continue

        # Create nested tabs
        ticker_tabs = []
        for ticker in ticker_results:
            model_tabs = []
            for model in ticker_results[ticker]:
                result = ticker_results[ticker][model]
                backtest_df = result['backtest_df']
                fig = self._create_backtest_plot(backtest_df, f"{ticker} ({model})")
                model_tab = dcc.Tab(label=model, children=[
                    dcc.Graph(figure=fig)
                ])
                model_tabs.append(model_tab)
            ticker_tab = dcc.Tab(label=ticker, children=[
                html.H4(f"{ticker} Models"),
                dcc.Tabs(id=f"{ticker}-model-tabs", children=model_tabs)
            ])
            ticker_tabs.append(ticker_tab)

        self.app.layout = html.Div([
            html.H1("Stock Price Forecasting Dashboard"),
            html.H3("Model Performance Metrics"),
            dash_table.DataTable(
                id='metrics-table',
                columns=[
                    {'name': 'Ticker', 'id': 'Ticker'},
                    {'name': 'Model', 'id': 'Model'},
                    {'name': 'Horizon', 'id': 'Horizon'},
                    {'name': 'RMSE', 'id': 'RMSE'},
                    {'name': 'MAE', 'id': 'MAE'},
                    {'name': 'R²', 'id': 'R²'}
                ],
                data=metrics_df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                sort_action='native',
                sort_mode='multi'
            ),
            html.H3("Backtest Results"),
            dcc.Tabs(id='ticker-tabs', children=ticker_tabs)
        ])

        # Save dashboard artifacts
        metrics_df.to_csv(os.path.join(self.output_dir, 'metrics_summary.csv'), index=False)
        mlflow.log_artifact(os.path.join(self.output_dir, 'metrics_summary.csv'))

        # Run dashboard
        logger.info("Starting Dash server for dashboard")
        try:
            self.app.run(debug=False, port=8050)
        except Exception as e:
            logger.error(f"Failed to run dashboard: {str(e)}")

    def _create_backtest_plot(self, backtest_df: pd.DataFrame, title: str) -> go.Figure:
        """
        Create Plotly figure for backtest results.
        
        Args:
            backtest_df (pd.DataFrame): Backtest data with Date, Actual, Predicted, Signal.
            title (str): Plot title (e.g., 'BTC-USD (XGBoost)').
        
        Returns:
            go.Figure: Plotly figure for backtest results.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=backtest_df['Date'],
            y=backtest_df['Actual'],
            mode='lines',
            name='Actual Price',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=backtest_df['Date'],
            y=backtest_df['Predicted'],
            mode='lines',
            name='Predicted Price',
            line=dict(color='orange', dash='dash')
        ))

        # Add Buy/Sell signals
        buy_signals = backtest_df[backtest_df['Signal'] == 'Buy']
        sell_signals = backtest_df[backtest_df['Signal'] == 'Sell']
        fig.add_trace(go.Scatter(
            x=buy_signals['Date'],
            y=buy_signals['Actual'],
            mode='markers',
            name='Buy Signal',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals['Date'],
            y=sell_signals['Actual'],
            mode='markers',
            name='Sell Signal',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))

        fig.update_layout(
            title=f"{title} Actual vs Predicted Prices",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            showlegend=True
        )

        # Save plot
        plot_path = os.path.join(self.output_dir, f"{title.replace(' ', '_').replace('(', '').replace(')', '')}_backtest_plot.html")
        fig.write_html(plot_path)
        mlflow.log_artifact(plot_path)

        return fig