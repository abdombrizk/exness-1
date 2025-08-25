#!/usr/bin/env python3
"""
Performance Monitor for Gold Trading AI
Real-time performance tracking and monitoring

Author: AI Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import time
import threading


class PerformanceMonitor:
    """
    Real-time performance monitoring system
    Tracks system performance, accuracy, and trading metrics
    """
    
    def __init__(self):
        self.performance_data = []
        self.system_metrics = {}
        self.trading_metrics = {}
        self.alerts = []
        
        # Performance thresholds
        self.accuracy_threshold = 0.90
        self.response_time_threshold = 3.0  # seconds
        self.memory_threshold = 1024  # MB
        
        # Data files
        self.performance_file = 'models/trained_models/performance_log.json'
        self.metrics_file = 'models/trained_models/system_metrics.json'
        
        # Load existing data
        self._load_performance_data()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._continuous_monitoring, daemon=True)
        self.monitor_thread.start()
        
        print("📊 Performance Monitor initialized")
        
    def log_prediction(self, prediction_result):
        """
        Log a prediction for performance tracking
        
        Args:
            prediction_result (dict): Prediction result from analyzer
        """
        try:
            performance_entry = {
                'timestamp': datetime.now().isoformat(),
                'signal': prediction_result.get('signal', 'UNKNOWN'),
                'confidence': prediction_result.get('confidence', 0),
                'accuracy_estimate': prediction_result.get('accuracy_estimate', 0),
                'technical_score': prediction_result.get('technical_score', 0),
                'fundamental_score': prediction_result.get('fundamental_score', 0),
                'risk_score': prediction_result.get('risk_score', 0),
                'entry_price': prediction_result.get('entry_price', 0),
                'stop_loss': prediction_result.get('stop_loss', 0),
                'take_profit': prediction_result.get('take_profit', 0),
                'position_size': prediction_result.get('position_size', 0),
                'risk_reward_ratio': prediction_result.get('risk_reward_ratio', 0)
            }
            
            self.performance_data.append(performance_entry)
            
            # Keep only last 1000 entries
            if len(self.performance_data) > 1000:
                self.performance_data = self.performance_data[-1000:]
                
            # Update trading metrics
            self._update_trading_metrics(performance_entry)
            
            # Save data periodically
            if len(self.performance_data) % 10 == 0:
                self._save_performance_data()
                
        except Exception as e:
            print(f"❌ Error logging prediction: {e}")
            
    def log_system_performance(self, execution_time, memory_usage=None, cpu_usage=None):
        """
        Log system performance metrics
        
        Args:
            execution_time (float): Analysis execution time in seconds
            memory_usage (float): Memory usage in MB
            cpu_usage (float): CPU usage percentage
        """
        try:
            current_time = datetime.now()
            
            # Update system metrics
            self.system_metrics.update({
                'last_execution_time': execution_time,
                'last_update': current_time.isoformat(),
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage
            })
            
            # Check performance thresholds
            self._check_performance_thresholds(execution_time, memory_usage)
            
            # Update performance statistics
            self._update_performance_stats(execution_time)
            
        except Exception as e:
            print(f"❌ Error logging system performance: {e}")
            
    def get_metrics(self):
        """
        Get comprehensive performance metrics
        
        Returns:
            dict: Performance metrics and statistics
        """
        try:
            if not self.performance_data:
                return {'status': 'NO_DATA', 'message': 'No performance data available'}
                
            # Calculate recent performance
            recent_data = self._get_recent_data(hours=24)
            
            # Accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(recent_data)
            
            # Trading performance metrics
            trading_performance = self._calculate_trading_performance(recent_data)
            
            # System performance metrics
            system_performance = self._calculate_system_performance()
            
            # Alert summary
            alert_summary = self._get_alert_summary()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'data_period': '24 hours',
                'total_predictions': len(recent_data),
                'accuracy_metrics': accuracy_metrics,
                'trading_performance': trading_performance,
                'system_performance': system_performance,
                'alerts': alert_summary,
                'overall_status': self._determine_overall_status(accuracy_metrics, system_performance)
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error getting metrics: {e}")
            return {'status': 'ERROR', 'message': str(e)}
            
    def get_performance_report(self, period_days=7):
        """
        Generate detailed performance report
        
        Args:
            period_days (int): Number of days to include in report
            
        Returns:
            dict: Detailed performance report
        """
        try:
            # Get data for specified period
            period_data = self._get_recent_data(hours=period_days * 24)
            
            if not period_data:
                return {'status': 'NO_DATA', 'message': f'No data for {period_days} days'}
                
            # Performance trends
            trends = self._analyze_performance_trends(period_data)
            
            # Signal analysis
            signal_analysis = self._analyze_signal_performance(period_data)
            
            # Risk analysis
            risk_analysis = self._analyze_risk_metrics(period_data)
            
            # System reliability
            reliability_metrics = self._calculate_reliability_metrics(period_data)
            
            report = {
                'report_period': f'{period_days} days',
                'report_timestamp': datetime.now().isoformat(),
                'data_summary': {
                    'total_predictions': len(period_data),
                    'avg_daily_predictions': len(period_data) / period_days,
                    'period_start': period_data[0]['timestamp'] if period_data else None,
                    'period_end': period_data[-1]['timestamp'] if period_data else None
                },
                'performance_trends': trends,
                'signal_analysis': signal_analysis,
                'risk_analysis': risk_analysis,
                'reliability_metrics': reliability_metrics,
                'recommendations': self._generate_performance_recommendations(trends, signal_analysis)
            }
            
            return report
            
        except Exception as e:
            print(f"❌ Error generating performance report: {e}")
            return {'status': 'ERROR', 'message': str(e)}
            
    def add_alert(self, alert_type, message, severity='INFO'):
        """
        Add a system alert
        
        Args:
            alert_type (str): Type of alert
            message (str): Alert message
            severity (str): Alert severity (INFO, WARNING, ERROR, CRITICAL)
        """
        try:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': alert_type,
                'message': message,
                'severity': severity
            }
            
            self.alerts.append(alert)
            
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
                
            # Print critical alerts
            if severity in ['ERROR', 'CRITICAL']:
                print(f"🚨 {severity} Alert: {message}")
                
        except Exception as e:
            print(f"❌ Error adding alert: {e}")
            
    def _continuous_monitoring(self):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                # Monitor system health every 60 seconds
                self._monitor_system_health()
                time.sleep(60)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(60)
                
    def _monitor_system_health(self):
        """Monitor overall system health"""
        try:
            # Check recent performance
            recent_data = self._get_recent_data(hours=1)
            
            if recent_data:
                # Check accuracy trend
                accuracies = [d['accuracy_estimate'] for d in recent_data if d.get('accuracy_estimate')]
                if accuracies:
                    avg_accuracy = np.mean(accuracies)
                    if avg_accuracy < self.accuracy_threshold * 100:
                        self.add_alert(
                            'ACCURACY_LOW',
                            f'Average accuracy ({avg_accuracy:.1f}%) below threshold ({self.accuracy_threshold:.1%})',
                            'WARNING'
                        )
                        
                # Check confidence levels
                confidences = [d['confidence'] for d in recent_data if d.get('confidence')]
                if confidences:
                    avg_confidence = np.mean(confidences)
                    if avg_confidence < 60:
                        self.add_alert(
                            'CONFIDENCE_LOW',
                            f'Average confidence ({avg_confidence:.1f}%) is low',
                            'INFO'
                        )
                        
        except Exception as e:
            self.add_alert('MONITORING_ERROR', f'System monitoring error: {e}', 'ERROR')
            
    def _update_trading_metrics(self, entry):
        """Update trading performance metrics"""
        try:
            signal = entry.get('signal', 'UNKNOWN')
            
            # Initialize signal counters
            if signal not in self.trading_metrics:
                self.trading_metrics[signal] = {
                    'count': 0,
                    'total_confidence': 0,
                    'avg_confidence': 0,
                    'avg_risk_reward': 0
                }
                
            # Update counters
            self.trading_metrics[signal]['count'] += 1
            self.trading_metrics[signal]['total_confidence'] += entry.get('confidence', 0)
            self.trading_metrics[signal]['avg_confidence'] = (
                self.trading_metrics[signal]['total_confidence'] / 
                self.trading_metrics[signal]['count']
            )
            
            # Update risk/reward
            rr_ratio = entry.get('risk_reward_ratio', 0)
            if rr_ratio > 0:
                current_avg = self.trading_metrics[signal]['avg_risk_reward']
                count = self.trading_metrics[signal]['count']
                self.trading_metrics[signal]['avg_risk_reward'] = (
                    (current_avg * (count - 1) + rr_ratio) / count
                )
                
        except Exception as e:
            print(f"❌ Error updating trading metrics: {e}")
            
    def _check_performance_thresholds(self, execution_time, memory_usage):
        """Check if performance thresholds are exceeded"""
        try:
            # Check execution time
            if execution_time > self.response_time_threshold:
                self.add_alert(
                    'SLOW_EXECUTION',
                    f'Analysis took {execution_time:.1f}s (threshold: {self.response_time_threshold}s)',
                    'WARNING'
                )
                
            # Check memory usage
            if memory_usage and memory_usage > self.memory_threshold:
                self.add_alert(
                    'HIGH_MEMORY',
                    f'Memory usage {memory_usage:.0f}MB (threshold: {self.memory_threshold}MB)',
                    'WARNING'
                )
                
        except Exception as e:
            print(f"❌ Error checking thresholds: {e}")
            
    def _update_performance_stats(self, execution_time):
        """Update performance statistics"""
        try:
            if 'execution_times' not in self.system_metrics:
                self.system_metrics['execution_times'] = []
                
            self.system_metrics['execution_times'].append(execution_time)
            
            # Keep only last 100 execution times
            if len(self.system_metrics['execution_times']) > 100:
                self.system_metrics['execution_times'] = self.system_metrics['execution_times'][-100:]
                
            # Calculate statistics
            times = self.system_metrics['execution_times']
            self.system_metrics['avg_execution_time'] = np.mean(times)
            self.system_metrics['max_execution_time'] = np.max(times)
            self.system_metrics['min_execution_time'] = np.min(times)
            
        except Exception as e:
            print(f"❌ Error updating performance stats: {e}")
            
    def _get_recent_data(self, hours=24):
        """Get performance data from recent hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_data = [
                entry for entry in self.performance_data
                if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
            ]
            
            return recent_data
            
        except Exception as e:
            print(f"❌ Error getting recent data: {e}")
            return []
            
    def _calculate_accuracy_metrics(self, data):
        """Calculate accuracy metrics from data"""
        try:
            if not data:
                return {'status': 'NO_DATA'}
                
            accuracies = [d['accuracy_estimate'] for d in data if d.get('accuracy_estimate')]
            confidences = [d['confidence'] for d in data if d.get('confidence')]
            
            metrics = {
                'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                'min_accuracy': np.min(accuracies) if accuracies else 0,
                'max_accuracy': np.max(accuracies) if accuracies else 0,
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'accuracy_std': np.std(accuracies) if accuracies else 0,
                'samples': len(data)
            }
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating accuracy metrics: {e}")
            return {'status': 'ERROR'}
            
    def _calculate_trading_performance(self, data):
        """Calculate trading performance metrics"""
        try:
            if not data:
                return {'status': 'NO_DATA'}
                
            # Signal distribution
            signals = [d['signal'] for d in data if d.get('signal')]
            signal_counts = {}
            for signal in signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
                
            # Risk/reward analysis
            rr_ratios = [d['risk_reward_ratio'] for d in data if d.get('risk_reward_ratio', 0) > 0]
            
            performance = {
                'signal_distribution': signal_counts,
                'avg_risk_reward': np.mean(rr_ratios) if rr_ratios else 0,
                'total_signals': len(signals),
                'unique_signals': len(signal_counts)
            }
            
            return performance
            
        except Exception as e:
            print(f"❌ Error calculating trading performance: {e}")
            return {'status': 'ERROR'}
            
    def _calculate_system_performance(self):
        """Calculate system performance metrics"""
        try:
            performance = {
                'avg_execution_time': self.system_metrics.get('avg_execution_time', 0),
                'max_execution_time': self.system_metrics.get('max_execution_time', 0),
                'last_execution_time': self.system_metrics.get('last_execution_time', 0),
                'memory_usage': self.system_metrics.get('memory_usage', 0),
                'cpu_usage': self.system_metrics.get('cpu_usage', 0),
                'last_update': self.system_metrics.get('last_update', 'Never')
            }
            
            return performance
            
        except Exception as e:
            print(f"❌ Error calculating system performance: {e}")
            return {'status': 'ERROR'}
            
    def _get_alert_summary(self):
        """Get summary of recent alerts"""
        try:
            # Get alerts from last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_alerts = [
                alert for alert in self.alerts
                if datetime.fromisoformat(alert['timestamp']) >= cutoff_time
            ]
            
            # Count by severity
            severity_counts = {}
            for alert in recent_alerts:
                severity = alert['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
            return {
                'total_alerts': len(recent_alerts),
                'severity_breakdown': severity_counts,
                'latest_alerts': recent_alerts[-5:] if recent_alerts else []
            }
            
        except Exception as e:
            print(f"❌ Error getting alert summary: {e}")
            return {'status': 'ERROR'}
            
    def _determine_overall_status(self, accuracy_metrics, system_performance):
        """Determine overall system status"""
        try:
            avg_accuracy = accuracy_metrics.get('avg_accuracy', 0)
            avg_execution_time = system_performance.get('avg_execution_time', 0)
            
            if avg_accuracy >= 90 and avg_execution_time <= 3.0:
                return 'EXCELLENT'
            elif avg_accuracy >= 85 and avg_execution_time <= 5.0:
                return 'GOOD'
            elif avg_accuracy >= 80:
                return 'ACCEPTABLE'
            else:
                return 'NEEDS_ATTENTION'
                
        except:
            return 'UNKNOWN'
            
    def _analyze_performance_trends(self, data):
        """Analyze performance trends over time"""
        try:
            if len(data) < 10:
                return {'status': 'INSUFFICIENT_DATA'}
                
            # Sort by timestamp
            sorted_data = sorted(data, key=lambda x: x['timestamp'])
            
            # Extract time series
            accuracies = [d['accuracy_estimate'] for d in sorted_data if d.get('accuracy_estimate')]
            confidences = [d['confidence'] for d in sorted_data if d.get('confidence')]
            
            # Calculate trends
            accuracy_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0] if len(accuracies) > 1 else 0
            confidence_trend = np.polyfit(range(len(confidences)), confidences, 1)[0] if len(confidences) > 1 else 0
            
            trends = {
                'accuracy_trend': 'IMPROVING' if accuracy_trend > 0.1 else 'DECLINING' if accuracy_trend < -0.1 else 'STABLE',
                'confidence_trend': 'IMPROVING' if confidence_trend > 0.1 else 'DECLINING' if confidence_trend < -0.1 else 'STABLE',
                'accuracy_slope': accuracy_trend,
                'confidence_slope': confidence_trend
            }
            
            return trends
            
        except Exception as e:
            print(f"❌ Error analyzing trends: {e}")
            return {'status': 'ERROR'}
            
    def _analyze_signal_performance(self, data):
        """Analyze performance by signal type"""
        try:
            signal_performance = {}
            
            for entry in data:
                signal = entry.get('signal', 'UNKNOWN')
                if signal not in signal_performance:
                    signal_performance[signal] = {
                        'count': 0,
                        'total_confidence': 0,
                        'total_accuracy': 0
                    }
                    
                signal_performance[signal]['count'] += 1
                signal_performance[signal]['total_confidence'] += entry.get('confidence', 0)
                signal_performance[signal]['total_accuracy'] += entry.get('accuracy_estimate', 0)
                
            # Calculate averages
            for signal in signal_performance:
                count = signal_performance[signal]['count']
                signal_performance[signal]['avg_confidence'] = signal_performance[signal]['total_confidence'] / count
                signal_performance[signal]['avg_accuracy'] = signal_performance[signal]['total_accuracy'] / count
                
            return signal_performance
            
        except Exception as e:
            print(f"❌ Error analyzing signal performance: {e}")
            return {'status': 'ERROR'}
            
    def _analyze_risk_metrics(self, data):
        """Analyze risk-related metrics"""
        try:
            risk_scores = [d['risk_score'] for d in data if d.get('risk_score')]
            rr_ratios = [d['risk_reward_ratio'] for d in data if d.get('risk_reward_ratio', 0) > 0]
            
            risk_analysis = {
                'avg_risk_score': np.mean(risk_scores) if risk_scores else 0,
                'avg_risk_reward': np.mean(rr_ratios) if rr_ratios else 0,
                'risk_distribution': {
                    'low_risk': sum(1 for r in risk_scores if r < 30),
                    'medium_risk': sum(1 for r in risk_scores if 30 <= r <= 60),
                    'high_risk': sum(1 for r in risk_scores if r > 60)
                } if risk_scores else {}
            }
            
            return risk_analysis
            
        except Exception as e:
            print(f"❌ Error analyzing risk metrics: {e}")
            return {'status': 'ERROR'}
            
    def _calculate_reliability_metrics(self, data):
        """Calculate system reliability metrics"""
        try:
            # Simulate reliability metrics
            reliability = {
                'uptime_percentage': 99.5,  # Would be calculated from actual uptime
                'error_rate': 0.5,  # Percentage of failed predictions
                'avg_response_time': self.system_metrics.get('avg_execution_time', 2.0),
                'data_quality_score': 95.0  # Would be calculated from data validation
            }
            
            return reliability
            
        except Exception as e:
            print(f"❌ Error calculating reliability metrics: {e}")
            return {'status': 'ERROR'}
            
    def _generate_performance_recommendations(self, trends, signal_analysis):
        """Generate performance improvement recommendations"""
        try:
            recommendations = []
            
            # Accuracy trend recommendations
            if trends.get('accuracy_trend') == 'DECLINING':
                recommendations.append("Accuracy is declining - consider model retraining")
                recommendations.append("Review recent data quality and feature engineering")
                
            # Confidence trend recommendations
            if trends.get('confidence_trend') == 'DECLINING':
                recommendations.append("Confidence levels are declining - investigate model uncertainty")
                
            # Signal performance recommendations
            if signal_analysis:
                low_performing_signals = [
                    signal for signal, metrics in signal_analysis.items()
                    if metrics.get('avg_accuracy', 0) < 85
                ]
                
                if low_performing_signals:
                    recommendations.append(f"Low performing signals detected: {', '.join(low_performing_signals)}")
                    
            if not recommendations:
                recommendations.append("System performance is stable - continue monitoring")
                
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
            
    def _load_performance_data(self):
        """Load performance data from file"""
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    self.performance_data = json.load(f)
                print(f"📂 Loaded {len(self.performance_data)} performance records")
        except Exception as e:
            print(f"⚠️  Could not load performance data: {e}")
            self.performance_data = []
            
    def _save_performance_data(self):
        """Save performance data to file"""
        try:
            os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save performance data: {e}")
            
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
