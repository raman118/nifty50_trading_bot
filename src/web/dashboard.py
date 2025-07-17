from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
import json
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from ..utils.logger import TradingLogger
from ..utils.config_manager import ConfigManager


class TradingDashboard:
    """Premium Purple Trading Dashboard"""

    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.logger = TradingLogger().get_logger()

        # Create Flask app
        self.app = Flask(__name__,
                         template_folder='templates',
                         static_folder='static')
        self.app.config['SECRET_KEY'] = 'nifty50_purple_dashboard'

        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Data storage for charts
        self.price_data = []
        self.signal_data = []
        self.portfolio_data = []

        # Setup routes and events
        self.setup_routes()
        self.setup_socket_events()
        self.start_background_tasks()

    def setup_routes(self):
        """Setup Flask routes with enhanced functionality"""

        @self.app.route('/')
        def index():
            return render_template('purple_dashboard.html')

        @self.app.route('/api/status')
        def get_status():
            """Get current bot status"""
            try:
                status = self.trading_bot.get_status()
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/portfolio')
        def get_portfolio():
            """Get portfolio summary"""
            try:
                portfolio = self.trading_bot.risk_manager.get_portfolio_summary()
                return jsonify(portfolio)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/signals')
        def get_signals():
            """Get recent signals"""
            try:
                if self.trading_bot.signal_generator:
                    signals = self.trading_bot.signal_generator.get_signal_history(20)
                    for signal in signals:
                        if 'timestamp' in signal:
                            signal['timestamp'] = signal['timestamp'].isoformat()
                    return jsonify(signals)
                return jsonify([])
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/chart/price')
        def get_price_chart():
            """Get purple-themed price chart"""
            try:
                recent_data = self.price_data[-100:] if self.price_data else []

                fig = go.Figure()

                if recent_data:
                    timestamps = [d['timestamp'] for d in recent_data]
                    prices = [d['price'] for d in recent_data]

                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=prices,
                        mode='lines+markers',
                        name='Nifty 50 Price',
                        line=dict(color='#8B5CF6', width=3),
                        marker=dict(color='#A855F7', size=4),
                        hovertemplate='<b>Price</b>: ₹%{y:,.2f}<br><b>Time</b>: %{x}<extra></extra>'
                    ))

                fig.update_layout(
                    title=dict(
                        text='🚀 Nifty 50 Live Price Movement',
                        font=dict(size=20, color='#6B46C1')
                    ),
                    xaxis_title='Time',
                    yaxis_title='Price (₹)',
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        gridcolor='#E5E7EB',
                        title_font=dict(color='#6B46C1')
                    ),
                    yaxis=dict(
                        gridcolor='#E5E7EB',
                        title_font=dict(color='#6B46C1')
                    ),
                    showlegend=False
                )

                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/chart/portfolio')
        def get_portfolio_chart():
            """Get purple-themed portfolio chart"""
            try:
                recent_data = self.portfolio_data[-50:] if self.portfolio_data else []

                fig = go.Figure()

                if recent_data:
                    timestamps = [d['timestamp'] for d in recent_data]
                    values = [d['portfolio_value'] for d in recent_data]

                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='#10B981', width=3),
                        fill='tonexty',
                        fillcolor='rgba(16, 185, 129, 0.1)',
                        hovertemplate='<b>Portfolio</b>: ₹%{y:,.2f}<br><b>Time</b>: %{x}<extra></extra>'
                    ))

                fig.update_layout(
                    title=dict(
                        text='📈 Portfolio Performance',
                        font=dict(size=20, color='#6B46C1')
                    ),
                    xaxis_title='Time',
                    yaxis_title='Value (₹)',
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        gridcolor='#E5E7EB',
                        title_font=dict(color='#6B46C1')
                    ),
                    yaxis=dict(
                        gridcolor='#E5E7EB',
                        title_font=dict(color='#6B46C1')
                    ),
                    showlegend=False
                )

                return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/control/<action>')
        def control_bot(action):
            """Control bot actions"""
            try:
                if action == 'start':
                    self.trading_bot.resume_trading()
                    return jsonify({'status': 'Trading started', 'action': 'success'})
                elif action == 'stop':
                    self.trading_bot.stop_trading()
                    return jsonify({'status': 'Trading paused', 'action': 'warning'})
                elif action == 'shutdown':
                    self.trading_bot.shutdown()
                    return jsonify({'status': 'Bot shutdown', 'action': 'danger'})
                else:
                    return jsonify({'error': 'Invalid action'}), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def setup_socket_events(self):
        """Setup WebSocket events for real-time updates"""

        @self.socketio.on('connect')
        def handle_connect():
            self.logger.info('Client connected to purple dashboard')
            emit('status', {'message': 'Connected to Nifty 50 Trading Bot', 'type': 'success'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info('Client disconnected from dashboard')

        @self.socketio.on('request_update')
        def handle_update_request():
            """Send current data to client"""
            try:
                status = self.trading_bot.get_status()
                emit('status_update', status)

                portfolio = self.trading_bot.risk_manager.get_portfolio_summary()
                emit('portfolio_update', portfolio)

            except Exception as e:
                emit('error', {'message': str(e)})

    def start_background_tasks(self):
        """Start background data collection and real-time updates"""

        def collect_and_broadcast():
            """Collect data and broadcast to clients"""
            while True:
                try:
                    # Collect price data
                    current_price = self.trading_bot.current_price
                    if current_price > 0:
                        self.price_data.append({
                            'timestamp': datetime.now().isoformat(),
                            'price': current_price
                        })

                    # Collect portfolio data
                    portfolio = self.trading_bot.risk_manager.get_portfolio_summary()
                    if portfolio:
                        self.portfolio_data.append({
                            'timestamp': datetime.now().isoformat(),
                            'portfolio_value': portfolio.get('portfolio_value', 0),
                            'daily_pnl': portfolio.get('daily_pnl', 0)
                        })

                    # Keep only last 1000 records
                    if len(self.price_data) > 1000:
                        self.price_data = self.price_data[-1000:]
                    if len(self.portfolio_data) > 1000:
                        self.portfolio_data = self.portfolio_data[-1000:]

                    # Broadcast real-time updates
                    self.socketio.emit('price_update', {
                        'price': current_price,
                        'timestamp': datetime.now().isoformat()
                    })

                    self.socketio.emit('portfolio_update', portfolio)

                    time.sleep(3)  # Update every 3 seconds

                except Exception as e:
                    self.logger.error(f"Error in data collection: {str(e)}")
                    time.sleep(10)

        # Start background thread
        data_thread = threading.Thread(target=collect_and_broadcast, daemon=True)
        data_thread.start()

    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the purple dashboard"""
        self.logger.info(f"🎨 Starting Purple Trading Dashboard on http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


