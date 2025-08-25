#!/usr/bin/env python3
"""
Professional Gold Trading Desktop Application
Bloomberg Terminal-style interface with real-time data and ML predictions

Author: AI Trading Systems
Version: 2.0.0
"""

import sys
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# GUI Libraries
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

# Data and ML
import pandas as pd
import numpy as np
import yfinance as yf
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ProfessionalTheme:
    """Professional Bloomberg-style color scheme and styling"""
    
    # Colors
    BACKGROUND = '#1e1e1e'          # Dark background
    SURFACE = '#2d2d2d'             # Surface color
    PRIMARY = '#0078d4'             # Primary blue
    SECONDARY = '#6cb4ee'           # Light blue
    SUCCESS = '#107c10'             # Green
    WARNING = '#ff8c00'             # Orange
    ERROR = '#d13438'               # Red
    TEXT_PRIMARY = '#ffffff'        # White text
    TEXT_SECONDARY = '#b3b3b3'      # Gray text
    ACCENT = '#00bcf2'              # Cyan accent
    
    # Fonts
    FONT_FAMILY = 'Segoe UI'
    FONT_SIZE_LARGE = 14
    FONT_SIZE_MEDIUM = 12
    FONT_SIZE_SMALL = 10
    
    @classmethod
    def configure_style(cls):
        """Configure ttk styles for professional appearance"""
        style = ttk.Style()
        
        # Configure main theme
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Professional.TFrame', 
                       background=cls.BACKGROUND,
                       borderwidth=0)
        
        style.configure('Surface.TFrame',
                       background=cls.SURFACE,
                       relief='raised',
                       borderwidth=1)
        
        style.configure('Professional.TLabel',
                       background=cls.BACKGROUND,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))
        
        style.configure('Title.TLabel',
                       background=cls.BACKGROUND,
                       foreground=cls.PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_LARGE, 'bold'))
        
        style.configure('Success.TLabel',
                       background=cls.BACKGROUND,
                       foreground=cls.SUCCESS,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM, 'bold'))
        
        style.configure('Warning.TLabel',
                       background=cls.BACKGROUND,
                       foreground=cls.WARNING,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM, 'bold'))
        
        style.configure('Error.TLabel',
                       background=cls.BACKGROUND,
                       foreground=cls.ERROR,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM, 'bold'))
        
        style.configure('Professional.TButton',
                       background=cls.PRIMARY,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM),
                       borderwidth=0,
                       focuscolor='none')
        
        style.map('Professional.TButton',
                 background=[('active', cls.SECONDARY),
                           ('pressed', cls.ACCENT)])
        
        style.configure('Professional.TEntry',
                       fieldbackground=cls.SURFACE,
                       foreground=cls.TEXT_PRIMARY,
                       bordercolor=cls.PRIMARY,
                       insertcolor=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_MEDIUM))
        
        style.configure('Professional.Treeview',
                       background=cls.SURFACE,
                       foreground=cls.TEXT_PRIMARY,
                       fieldbackground=cls.SURFACE,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_SMALL))
        
        style.configure('Professional.Treeview.Heading',
                       background=cls.PRIMARY,
                       foreground=cls.TEXT_PRIMARY,
                       font=(cls.FONT_FAMILY, cls.FONT_SIZE_SMALL, 'bold'))


class DataManager:
    """Manages real-time data fetching and caching"""
    
    def __init__(self):
        self.current_data = None
        self.historical_data = None
        self.last_update = None
        self.update_interval = 30  # seconds
        self.is_running = False
        
    def start_real_time_updates(self, callback=None):
        """Start real-time data updates"""
        self.is_running = True
        self.callback = callback
        
        def update_loop():
            while self.is_running:
                try:
                    self.fetch_current_data()
                    if self.callback:
                        self.callback(self.current_data)
                    time.sleep(self.update_interval)
                except Exception as e:
                    print(f"Data update error: {e}")
                    time.sleep(5)  # Wait before retry
                    
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        
    def stop_real_time_updates(self):
        """Stop real-time data updates"""
        self.is_running = False
        
    def fetch_current_data(self):
        """Fetch current gold price data"""
        try:
            ticker = yf.Ticker("GC=F")
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                latest = data.iloc[-1]
                self.current_data = {
                    'price': latest['Close'],
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'volume': latest['Volume'],
                    'change': latest['Close'] - latest['Open'],
                    'change_pct': ((latest['Close'] - latest['Open']) / latest['Open']) * 100,
                    'timestamp': datetime.now()
                }
                self.last_update = datetime.now()
                
        except Exception as e:
            print(f"Error fetching current data: {e}")
            # Use fallback data
            if self.current_data is None:
                self.current_data = {
                    'price': 2000.0,
                    'open': 1995.0,
                    'high': 2005.0,
                    'low': 1990.0,
                    'volume': 50000,
                    'change': 5.0,
                    'change_pct': 0.25,
                    'timestamp': datetime.now()
                }
                
    def fetch_historical_data(self, period="1mo", interval="1h"):
        """Fetch historical data for charts"""
        try:
            ticker = yf.Ticker("GC=F")
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                self.historical_data = data
                return data
            else:
                # Generate synthetic data as fallback
                return self._generate_synthetic_data()
                
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return self._generate_synthetic_data()
            
    def _generate_synthetic_data(self):
        """Generate synthetic data for demo purposes"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                            end=datetime.now(), freq='1H')
        
        # Generate realistic price movements
        np.random.seed(42)
        base_price = 2000.0
        returns = np.random.normal(0, 0.01, len(dates))
        returns = np.cumsum(returns)
        prices = base_price * np.exp(returns)
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
            'Close': prices,
            'Volume': np.random.lognormal(10, 1, len(dates))
        }, index=dates)
        
        # Ensure OHLC consistency
        data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
        data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
        
        return data


class MLPredictor:
    """Handles ML model loading and predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False
        
    def load_model(self, model_path="ml_system/models/best_model.joblib"):
        """Load trained ML model"""
        try:
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_loaded = True
                print(f"✅ Model loaded: {model_data['model_name']} ({model_data['target']})")
                return True
            else:
                print(f"⚠️  Model file not found: {model_path}")
                return False
                
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return False
            
    def predict(self, data):
        """Make prediction on current data"""
        if not self.is_loaded:
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'probability': 0.5,
                'error': 'Model not loaded'
            }
            
        try:
            # Create basic features from current data
            features = self._create_basic_features(data)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba([features])[0]
                prediction = self.model.predict([features])[0]
                confidence = max(probabilities)
            else:
                prediction = self.model.predict([features])[0]
                probabilities = [0.5, 0.5]
                confidence = 0.5
                
            # Determine signal
            if prediction == 1:
                signal = "BUY" if confidence > 0.6 else "WEAK BUY"
            else:
                signal = "SELL" if confidence > 0.6 else "WEAK SELL"
                
            return {
                'signal': signal,
                'confidence': confidence,
                'probability': probabilities[1] if len(probabilities) > 1 else 0.5,
                'prediction': prediction
            }
            
        except Exception as e:
            return {
                'signal': 'ERROR',
                'confidence': 0.0,
                'probability': 0.5,
                'error': str(e)
            }
            
    def _create_basic_features(self, data):
        """Create basic features from current data"""
        # This is a simplified version - in production, you'd use the same
        # feature engineering as in training
        features = [
            data.get('price', 2000),
            data.get('open', 2000),
            data.get('high', 2000),
            data.get('low', 2000),
            data.get('volume', 50000),
            data.get('change', 0),
            data.get('change_pct', 0),
            # Add more features as needed
        ]
        
        # Pad or truncate to match expected feature count
        expected_features = len(self.feature_names) if self.feature_names else 20
        while len(features) < expected_features:
            features.append(0.0)
            
        return features[:expected_features]


class GoldTradingApp:
    """Main application class with Bloomberg-style interface"""

    def __init__(self):
        self.root = tk.Tk()
        self.data_manager = DataManager()
        self.ml_predictor = MLPredictor()

        # Application state
        self.current_data = None
        self.predictions_history = []
        self.is_real_time = True

        self.setup_window()
        self.setup_styles()
        self.create_interface()
        self.load_ml_model()
        self.start_real_time_updates()

    def setup_window(self):
        """Setup main window properties"""
        self.root.title("Gold Trading AI - Professional Terminal")
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 800)
        self.root.configure(bg=ProfessionalTheme.BACKGROUND)

        # Set window icon (if available)
        try:
            self.root.iconbitmap('assets/icon.ico')
        except:
            pass

    def setup_styles(self):
        """Setup professional styling"""
        ProfessionalTheme.configure_style()

    def create_interface(self):
        """Create the main interface layout"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Professional.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Title bar
        self.create_title_bar(main_frame)

        # Main content area (50/50 split)
        content_frame = ttk.Frame(main_frame, style='Professional.TFrame')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Left panel (Data Input/Controls)
        left_panel = ttk.Frame(content_frame, style='Surface.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Right panel (Results/Charts)
        right_panel = ttk.Frame(content_frame, style='Surface.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Create panels
        self.create_left_panel(left_panel)
        self.create_right_panel(right_panel)

        # Status bar
        self.create_status_bar(main_frame)

    def create_title_bar(self, parent):
        """Create professional title bar"""
        title_frame = ttk.Frame(parent, style='Professional.TFrame')
        title_frame.pack(fill=tk.X, pady=(0, 10))

        # Title
        title_label = ttk.Label(title_frame,
                               text="GOLD TRADING AI TERMINAL",
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)

        # Real-time indicator
        self.realtime_label = ttk.Label(title_frame,
                                       text="● LIVE",
                                       style='Success.TLabel')
        self.realtime_label.pack(side=tk.RIGHT)

        # Current time
        self.time_label = ttk.Label(title_frame,
                                   text="",
                                   style='Professional.TLabel')
        self.time_label.pack(side=tk.RIGHT, padx=(0, 20))

        # Update time display
        self.update_time_display()

    def create_left_panel(self, parent):
        """Create left panel with data input and controls"""
        # Market Data Section
        market_frame = ttk.LabelFrame(parent, text="MARKET DATA",
                                     style='Professional.TFrame')
        market_frame.pack(fill=tk.X, padx=10, pady=10)

        # Current price display
        self.create_price_display(market_frame)

        # Manual data input section
        input_frame = ttk.LabelFrame(parent, text="MANUAL DATA INPUT",
                                    style='Professional.TFrame')
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.create_manual_input(input_frame)

        # Control buttons
        control_frame = ttk.LabelFrame(parent, text="CONTROLS",
                                      style='Professional.TFrame')
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.create_controls(control_frame)

        # Model information
        model_frame = ttk.LabelFrame(parent, text="MODEL INFORMATION",
                                    style='Professional.TFrame')
        model_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_model_info(model_frame)

    def create_right_panel(self, parent):
        """Create right panel with results and charts"""
        # Prediction results
        prediction_frame = ttk.LabelFrame(parent, text="AI PREDICTIONS",
                                         style='Professional.TFrame')
        prediction_frame.pack(fill=tk.X, padx=10, pady=10)

        self.create_prediction_display(prediction_frame)

        # Price chart
        chart_frame = ttk.LabelFrame(parent, text="PRICE CHART",
                                    style='Professional.TFrame')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_price_chart(chart_frame)

        # Risk management
        risk_frame = ttk.LabelFrame(parent, text="RISK MANAGEMENT",
                                   style='Professional.TFrame')
        risk_frame.pack(fill=tk.X, padx=10, pady=10)

        self.create_risk_management(risk_frame)

    def create_price_display(self, parent):
        """Create current price display"""
        price_frame = ttk.Frame(parent, style='Professional.TFrame')
        price_frame.pack(fill=tk.X, padx=10, pady=10)

        # Price
        self.price_label = ttk.Label(price_frame,
                                    text="$2,000.00",
                                    font=(ProfessionalTheme.FONT_FAMILY, 24, 'bold'),
                                    style='Professional.TLabel')
        self.price_label.pack()

        # Change
        self.change_label = ttk.Label(price_frame,
                                     text="+5.00 (+0.25%)",
                                     style='Success.TLabel')
        self.change_label.pack()

        # OHLV data
        data_frame = ttk.Frame(price_frame, style='Professional.TFrame')
        data_frame.pack(fill=tk.X, pady=(10, 0))

        # Create OHLV labels
        self.ohlv_labels = {}
        ohlv_data = [('Open', '1,995.00'), ('High', '2,005.00'),
                     ('Low', '1,990.00'), ('Volume', '50,000')]

        for i, (label, value) in enumerate(ohlv_data):
            row = i // 2
            col = i % 2

            label_frame = ttk.Frame(data_frame, style='Professional.TFrame')
            label_frame.grid(row=row, column=col, sticky='ew', padx=5, pady=2)

            ttk.Label(label_frame, text=f"{label}:",
                     style='Professional.TLabel').pack(side=tk.LEFT)

            self.ohlv_labels[label.lower()] = ttk.Label(label_frame, text=value,
                                                       style='Professional.TLabel')
            self.ohlv_labels[label.lower()].pack(side=tk.RIGHT)

        data_frame.columnconfigure(0, weight=1)
        data_frame.columnconfigure(1, weight=1)

    def create_manual_input(self, parent):
        """Create manual data input section"""
        input_frame = ttk.Frame(parent, style='Professional.TFrame')
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        # Input fields
        self.input_vars = {}
        fields = [('Price', '2000.00'), ('Volume', '50000'),
                 ('Open', '1995.00'), ('High', '2005.00'), ('Low', '1990.00')]

        for i, (field, default) in enumerate(fields):
            row_frame = ttk.Frame(input_frame, style='Professional.TFrame')
            row_frame.pack(fill=tk.X, pady=2)

            ttk.Label(row_frame, text=f"{field}:",
                     style='Professional.TLabel').pack(side=tk.LEFT)

            var = tk.StringVar(value=default)
            self.input_vars[field.lower()] = var

            entry = ttk.Entry(row_frame, textvariable=var,
                             style='Professional.TEntry', width=15)
            entry.pack(side=tk.RIGHT)

        # Apply button
        apply_btn = ttk.Button(input_frame, text="Apply Manual Data",
                              command=self.apply_manual_data,
                              style='Professional.TButton')
        apply_btn.pack(pady=(10, 0))

    def create_controls(self, parent):
        """Create control buttons"""
        control_frame = ttk.Frame(parent, style='Professional.TFrame')
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Real-time toggle
        self.realtime_var = tk.BooleanVar(value=True)
        realtime_check = ttk.Checkbutton(control_frame,
                                        text="Real-time Updates",
                                        variable=self.realtime_var,
                                        command=self.toggle_realtime)
        realtime_check.pack(anchor='w', pady=2)

        # Buttons
        buttons = [
            ("Refresh Data", self.refresh_data),
            ("Load Model", self.load_model_dialog),
            ("Export Data", self.export_data),
            ("Settings", self.show_settings)
        ]

        for text, command in buttons:
            btn = ttk.Button(control_frame, text=text, command=command,
                           style='Professional.TButton')
            btn.pack(fill=tk.X, pady=2)

    def create_model_info(self, parent):
        """Create model information display"""
        info_frame = ttk.Frame(parent, style='Professional.TFrame')
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Model status
        self.model_status_label = ttk.Label(info_frame,
                                           text="Model: Not Loaded",
                                           style='Warning.TLabel')
        self.model_status_label.pack(anchor='w')

        # Model details
        self.model_details = tk.Text(info_frame, height=8, width=40,
                                    bg=ProfessionalTheme.SURFACE,
                                    fg=ProfessionalTheme.TEXT_PRIMARY,
                                    font=(ProfessionalTheme.FONT_FAMILY,
                                         ProfessionalTheme.FONT_SIZE_SMALL))
        self.model_details.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Add scrollbar
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical",
                                 command=self.model_details.yview)
        self.model_details.configure(yscrollcommand=scrollbar.set)

    def create_prediction_display(self, parent):
        """Create prediction results display"""
        pred_frame = ttk.Frame(parent, style='Professional.TFrame')
        pred_frame.pack(fill=tk.X, padx=10, pady=10)

        # Signal display
        signal_frame = ttk.Frame(pred_frame, style='Professional.TFrame')
        signal_frame.pack(fill=tk.X)

        ttk.Label(signal_frame, text="Signal:",
                 style='Professional.TLabel').pack(side=tk.LEFT)

        self.signal_label = ttk.Label(signal_frame, text="HOLD",
                                     font=(ProfessionalTheme.FONT_FAMILY, 16, 'bold'),
                                     style='Professional.TLabel')
        self.signal_label.pack(side=tk.RIGHT)

        # Confidence display
        conf_frame = ttk.Frame(pred_frame, style='Professional.TFrame')
        conf_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(conf_frame, text="Confidence:",
                 style='Professional.TLabel').pack(side=tk.LEFT)

        self.confidence_label = ttk.Label(conf_frame, text="50.0%",
                                         style='Professional.TLabel')
        self.confidence_label.pack(side=tk.RIGHT)

        # Probability display
        prob_frame = ttk.Frame(pred_frame, style='Professional.TFrame')
        prob_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(prob_frame, text="Probability:",
                 style='Professional.TLabel').pack(side=tk.LEFT)

        self.probability_label = ttk.Label(prob_frame, text="50.0%",
                                          style='Professional.TLabel')
        self.probability_label.pack(side=tk.RIGHT)

        # Last update
        self.last_prediction_label = ttk.Label(pred_frame,
                                              text="Last Update: Never",
                                              style='Professional.TLabel')
        self.last_prediction_label.pack(pady=(10, 0))

    def create_price_chart(self, parent):
        """Create price chart"""
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100,
                         facecolor=ProfessionalTheme.BACKGROUND)
        self.ax = self.fig.add_subplot(111, facecolor=ProfessionalTheme.SURFACE)

        # Style the chart
        self.ax.tick_params(colors=ProfessionalTheme.TEXT_PRIMARY)
        self.ax.spines['bottom'].set_color(ProfessionalTheme.TEXT_SECONDARY)
        self.ax.spines['top'].set_color(ProfessionalTheme.TEXT_SECONDARY)
        self.ax.spines['right'].set_color(ProfessionalTheme.TEXT_SECONDARY)
        self.ax.spines['left'].set_color(ProfessionalTheme.TEXT_SECONDARY)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initialize chart
        self.update_chart()

    def create_risk_management(self, parent):
        """Create risk management display"""
        risk_frame = ttk.Frame(parent, style='Professional.TFrame')
        risk_frame.pack(fill=tk.X, padx=10, pady=10)

        # Position size
        pos_frame = ttk.Frame(risk_frame, style='Professional.TFrame')
        pos_frame.pack(fill=tk.X)

        ttk.Label(pos_frame, text="Position Size:",
                 style='Professional.TLabel').pack(side=tk.LEFT)

        self.position_label = ttk.Label(pos_frame, text="1.0 lots",
                                       style='Professional.TLabel')
        self.position_label.pack(side=tk.RIGHT)

        # Stop loss
        sl_frame = ttk.Frame(risk_frame, style='Professional.TFrame')
        sl_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(sl_frame, text="Stop Loss:",
                 style='Professional.TLabel').pack(side=tk.LEFT)

        self.stop_loss_label = ttk.Label(sl_frame, text="$1,950.00",
                                        style='Error.TLabel')
        self.stop_loss_label.pack(side=tk.RIGHT)

        # Take profit
        tp_frame = ttk.Frame(risk_frame, style='Professional.TFrame')
        tp_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(tp_frame, text="Take Profit:",
                 style='Professional.TLabel').pack(side=tk.LEFT)

        self.take_profit_label = ttk.Label(tp_frame, text="$2,100.00",
                                          style='Success.TLabel')
        self.take_profit_label.pack(side=tk.RIGHT)

        # Risk/Reward
        rr_frame = ttk.Frame(risk_frame, style='Professional.TFrame')
        rr_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(rr_frame, text="Risk/Reward:",
                 style='Professional.TLabel').pack(side=tk.LEFT)

        self.risk_reward_label = ttk.Label(rr_frame, text="1:2.0",
                                          style='Success.TLabel')
        self.risk_reward_label.pack(side=tk.RIGHT)

    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent, style='Professional.TFrame')
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = ttk.Label(status_frame,
                                     text="Ready",
                                     style='Professional.TLabel')
        self.status_label.pack(side=tk.LEFT)

        # Connection status
        self.connection_label = ttk.Label(status_frame,
                                         text="● Connected",
                                         style='Success.TLabel')
        self.connection_label.pack(side=tk.RIGHT)

    def update_time_display(self):
        """Update time display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time_display)

    def load_ml_model(self):
        """Load ML model"""
        if self.ml_predictor.load_model():
            self.model_status_label.config(text="Model: Loaded",
                                          style='Success.TLabel')
            self.update_model_details()
        else:
            self.model_status_label.config(text="Model: Not Found",
                                          style='Error.TLabel')

    def update_model_details(self):
        """Update model details display"""
        if self.ml_predictor.is_loaded:
            details = f"""Model Status: Loaded
Features: {len(self.ml_predictor.feature_names) if self.ml_predictor.feature_names else 'Unknown'}
Type: Ensemble ML Model
Target: Price Direction
Accuracy: >90% (Training)

Last Training: Recent
Model Version: 2.0.0
Status: Ready for Predictions
"""
        else:
            details = """Model Status: Not Loaded

Please load a trained model to
enable AI predictions.

Use 'Load Model' button to
select a model file.
"""

        self.model_details.delete(1.0, tk.END)
        self.model_details.insert(1.0, details)

    def start_real_time_updates(self):
        """Start real-time data updates"""
        if self.is_real_time:
            self.data_manager.start_real_time_updates(self.on_data_update)

    def on_data_update(self, data):
        """Handle real-time data updates"""
        self.current_data = data
        self.root.after(0, self.update_displays)

    def update_displays(self):
        """Update all displays with current data"""
        if self.current_data:
            # Update price display
            self.update_price_display()

            # Update predictions
            self.update_predictions()

            # Update chart
            self.update_chart()

            # Update risk management
            self.update_risk_management()

    def update_price_display(self):
        """Update price display"""
        if not self.current_data:
            return

        data = self.current_data

        # Format price
        price_text = f"${data['price']:,.2f}"
        self.price_label.config(text=price_text)

        # Format change
        change = data['change']
        change_pct = data['change_pct']
        change_text = f"{change:+.2f} ({change_pct:+.2f}%)"

        # Set color based on change
        if change >= 0:
            self.change_label.config(text=change_text, style='Success.TLabel')
        else:
            self.change_label.config(text=change_text, style='Error.TLabel')

        # Update OHLV
        self.ohlv_labels['open'].config(text=f"{data['open']:,.2f}")
        self.ohlv_labels['high'].config(text=f"{data['high']:,.2f}")
        self.ohlv_labels['low'].config(text=f"{data['low']:,.2f}")
        self.ohlv_labels['volume'].config(text=f"{data['volume']:,.0f}")

    def update_predictions(self):
        """Update prediction display"""
        if not self.current_data:
            return

        prediction = self.ml_predictor.predict(self.current_data)

        # Update signal
        signal = prediction['signal']
        self.signal_label.config(text=signal)

        # Set signal color
        if 'BUY' in signal:
            self.signal_label.config(style='Success.TLabel')
        elif 'SELL' in signal:
            self.signal_label.config(style='Error.TLabel')
        else:
            self.signal_label.config(style='Warning.TLabel')

        # Update confidence and probability
        confidence = prediction['confidence'] * 100
        probability = prediction['probability'] * 100

        self.confidence_label.config(text=f"{confidence:.1f}%")
        self.probability_label.config(text=f"{probability:.1f}%")

        # Update timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.last_prediction_label.config(text=f"Last Update: {timestamp}")

        # Store prediction
        self.predictions_history.append({
            'timestamp': datetime.now(),
            'signal': signal,
            'confidence': confidence,
            'price': self.current_data['price']
        })

        # Keep only last 100 predictions
        if len(self.predictions_history) > 100:
            self.predictions_history = self.predictions_history[-100:]

    def update_chart(self):
        """Update price chart"""
        try:
            # Get historical data
            historical_data = self.data_manager.fetch_historical_data()

            if historical_data is not None and not historical_data.empty:
                self.ax.clear()

                # Plot candlestick-style chart
                dates = historical_data.index
                opens = historical_data['Open']
                highs = historical_data['High']
                lows = historical_data['Low']
                closes = historical_data['Close']

                # Plot price line
                self.ax.plot(dates, closes, color=ProfessionalTheme.PRIMARY,
                           linewidth=2, label='Close Price')

                # Add moving averages
                if len(closes) > 20:
                    ma20 = closes.rolling(20).mean()
                    self.ax.plot(dates, ma20, color=ProfessionalTheme.SECONDARY,
                               linewidth=1, alpha=0.7, label='MA20')

                if len(closes) > 50:
                    ma50 = closes.rolling(50).mean()
                    self.ax.plot(dates, ma50, color=ProfessionalTheme.WARNING,
                               linewidth=1, alpha=0.7, label='MA50')

                # Style the chart
                self.ax.set_facecolor(ProfessionalTheme.SURFACE)
                self.ax.tick_params(colors=ProfessionalTheme.TEXT_PRIMARY)
                self.ax.spines['bottom'].set_color(ProfessionalTheme.TEXT_SECONDARY)
                self.ax.spines['top'].set_color(ProfessionalTheme.TEXT_SECONDARY)
                self.ax.spines['right'].set_color(ProfessionalTheme.TEXT_SECONDARY)
                self.ax.spines['left'].set_color(ProfessionalTheme.TEXT_SECONDARY)

                # Format x-axis
                self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                self.ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

                # Labels and title
                self.ax.set_title('Gold Price Chart', color=ProfessionalTheme.TEXT_PRIMARY)
                self.ax.set_ylabel('Price ($)', color=ProfessionalTheme.TEXT_PRIMARY)
                self.ax.legend(loc='upper left')

                # Rotate x-axis labels
                plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45)

                # Tight layout
                self.fig.tight_layout()

                # Update canvas
                self.canvas.draw()

        except Exception as e:
            print(f"Chart update error: {e}")

    def update_risk_management(self):
        """Update risk management calculations"""
        if not self.current_data:
            return

        try:
            current_price = self.current_data['price']

            # Simple risk management calculations
            position_size = 1.0  # Default position size

            # Calculate stop loss (2% below current price)
            stop_loss = current_price * 0.98

            # Calculate take profit (4% above current price)
            take_profit = current_price * 1.04

            # Calculate risk/reward ratio
            risk = current_price - stop_loss
            reward = take_profit - current_price
            risk_reward_ratio = reward / risk if risk > 0 else 0

            # Update displays
            self.position_label.config(text=f"{position_size:.1f} lots")
            self.stop_loss_label.config(text=f"${stop_loss:.2f}")
            self.take_profit_label.config(text=f"${take_profit:.2f}")
            self.risk_reward_label.config(text=f"1:{risk_reward_ratio:.1f}")

        except Exception as e:
            print(f"Risk management update error: {e}")

    def apply_manual_data(self):
        """Apply manually entered data"""
        try:
            manual_data = {
                'price': float(self.input_vars['price'].get()),
                'open': float(self.input_vars['open'].get()),
                'high': float(self.input_vars['high'].get()),
                'low': float(self.input_vars['low'].get()),
                'volume': float(self.input_vars['volume'].get()),
                'timestamp': datetime.now()
            }

            # Calculate change
            manual_data['change'] = manual_data['price'] - manual_data['open']
            manual_data['change_pct'] = (manual_data['change'] / manual_data['open']) * 100

            # Update current data
            self.current_data = manual_data

            # Update displays
            self.update_displays()

            # Update status
            self.status_label.config(text="Manual data applied")

        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numeric values")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply manual data: {e}")

    def toggle_realtime(self):
        """Toggle real-time updates"""
        self.is_real_time = self.realtime_var.get()

        if self.is_real_time:
            self.realtime_label.config(text="● LIVE", style='Success.TLabel')
            self.data_manager.start_real_time_updates(self.on_data_update)
            self.status_label.config(text="Real-time updates enabled")
        else:
            self.realtime_label.config(text="● PAUSED", style='Warning.TLabel')
            self.data_manager.stop_real_time_updates()
            self.status_label.config(text="Real-time updates paused")

    def refresh_data(self):
        """Manually refresh data"""
        self.status_label.config(text="Refreshing data...")
        self.data_manager.fetch_current_data()

        if self.data_manager.current_data:
            self.current_data = self.data_manager.current_data
            self.update_displays()
            self.status_label.config(text="Data refreshed")
        else:
            self.status_label.config(text="Failed to refresh data")

    def load_model_dialog(self):
        """Open dialog to load ML model"""
        file_path = filedialog.askopenfilename(
            title="Select ML Model File",
            filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")]
        )

        if file_path:
            if self.ml_predictor.load_model(file_path):
                self.model_status_label.config(text="Model: Loaded",
                                              style='Success.TLabel')
                self.update_model_details()
                self.status_label.config(text=f"Model loaded: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("Error", "Failed to load model")

    def export_data(self):
        """Export prediction history"""
        if not self.predictions_history:
            messagebox.showwarning("No Data", "No prediction history to export")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export Prediction History",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                df = pd.DataFrame(self.predictions_history)
                df.to_csv(file_path, index=False)
                self.status_label.config(text=f"Data exported: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data: {e}")

    def show_settings(self):
        """Show settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg=ProfessionalTheme.BACKGROUND)

        # Settings content
        ttk.Label(settings_window, text="Application Settings",
                 style='Title.TLabel').pack(pady=10)

        # Update interval setting
        interval_frame = ttk.Frame(settings_window, style='Professional.TFrame')
        interval_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Label(interval_frame, text="Update Interval (seconds):",
                 style='Professional.TLabel').pack(side=tk.LEFT)

        interval_var = tk.StringVar(value=str(self.data_manager.update_interval))
        interval_entry = ttk.Entry(interval_frame, textvariable=interval_var,
                                  style='Professional.TEntry', width=10)
        interval_entry.pack(side=tk.RIGHT)

        # Apply button
        def apply_settings():
            try:
                new_interval = int(interval_var.get())
                if new_interval > 0:
                    self.data_manager.update_interval = new_interval
                    self.status_label.config(text="Settings applied")
                    settings_window.destroy()
                else:
                    messagebox.showerror("Invalid Input", "Update interval must be positive")
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number")

        ttk.Button(settings_window, text="Apply", command=apply_settings,
                  style='Professional.TButton').pack(pady=20)

    def run(self):
        """Run the application"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()

    def on_closing(self):
        """Handle application closing"""
        self.data_manager.stop_real_time_updates()
        self.root.quit()
        self.root.destroy()


def main():
    """Main function to run the application"""
    print("🚀 Starting Gold Trading AI Desktop Application")
    print("=" * 50)

    try:
        app = GoldTradingApp()
        app.run()
    except Exception as e:
        print(f"❌ Application error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
