#!/usr/bin/env python3
"""
Risk Management System for Gold Trading
Advanced risk management with position sizing and drawdown control

Author: AI Trading Systems
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class RiskManager:
    """
    Advanced risk management system for gold trading
    Implements position sizing, stop-loss optimization, and drawdown control
    """
    
    def __init__(self, max_risk_per_trade=0.02, max_portfolio_risk=0.10, max_drawdown=0.05):
        # Risk parameters
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_portfolio_risk = max_portfolio_risk  # 10% total portfolio risk
        self.max_drawdown = max_drawdown  # 5% maximum drawdown
        
        # Portfolio tracking
        self.portfolio_value = 100000  # Default $100k portfolio
        self.current_positions = []
        self.trade_history = []
        self.drawdown_history = []
        
        # Risk metrics
        self.current_drawdown = 0.0
        self.peak_portfolio_value = self.portfolio_value
        self.total_risk_exposure = 0.0
        
        print("⚖️ Risk Manager initialized")
        print(f"   Max risk per trade: {max_risk_per_trade:.1%}")
        print(f"   Max portfolio risk: {max_portfolio_risk:.1%}")
        print(f"   Max drawdown: {max_drawdown:.1%}")
        
    def calculate_position_size(self, account_balance=None, entry_price=None, stop_loss=None, confidence=None, volatility=None):
        """
        Calculate optimal position size based on risk parameters

        Args:
            account_balance (float): Account balance for position sizing
            entry_price (float): Entry price for the trade
            stop_loss (float): Stop loss price
            confidence (float): Model confidence (0-1 or 0-100)
            volatility (float): Current market volatility

        Returns:
            float: Position size recommendation
        """
        try:
            # Validate inputs
            if account_balance is None:
                account_balance = 10000  # Default account balance
            if entry_price is None or stop_loss is None:
                return 0.1  # Conservative default position size
            if confidence is None:
                confidence = 0.5

            # Normalize confidence to 0-1 range
            if confidence > 1:
                confidence = confidence / 100

            # Calculate risk per share/unit
            risk_per_unit = abs(entry_price - stop_loss)

            if risk_per_unit <= 0:
                return 0.1  # Conservative default

            # Base position size calculation using account balance
            base_risk_amount = account_balance * self.max_risk_per_trade
            base_position_size = base_risk_amount / risk_per_unit
            
            # Adjust for confidence
            confidence_factor = self._calculate_confidence_factor(confidence)
            
            # Adjust for volatility
            volatility_factor = self._calculate_volatility_factor(volatility)
            
            # Adjust for current portfolio risk
            portfolio_risk_factor = self._calculate_portfolio_risk_factor()
            
            # Adjust for drawdown
            drawdown_factor = self._calculate_drawdown_factor()
            
            # Calculate final position size
            adjusted_position_size = (
                base_position_size * 
                confidence_factor * 
                volatility_factor * 
                portfolio_risk_factor * 
                drawdown_factor
            )
            
            # Apply maximum position limits
            max_position_value = account_balance * 0.20  # Max 20% in single position
            max_position_size = max_position_value / entry_price

            final_position_size = min(adjusted_position_size, max_position_size)
            final_position_size = max(0.01, final_position_size)  # Minimum position

            # Return simple position size as expected by tests
            return round(final_position_size, 2)

        except Exception as e:
            print(f"❌ Position sizing error: {e}")
            return 0.1  # Conservative default

    def _get_position_recommendation(self, risk_percentage, confidence):
        """Get position recommendation based on risk and confidence"""
        if risk_percentage > 0.03:
            return "HIGH RISK - Consider reducing position size"
        elif risk_percentage > 0.02:
            return "MODERATE RISK - Acceptable with good confidence"
        elif confidence > 80:
            return "LOW RISK - Good opportunity with high confidence"
        else:
            return "CONSERVATIVE - Standard position sizing recommended"
            
    def optimize_stop_loss(self, entry_price, signal_direction, volatility, technical_levels=None):
        """
        Optimize stop-loss placement based on market conditions
        
        Args:
            entry_price (float): Entry price
            signal_direction (str): BUY or SELL
            volatility (float): Current market volatility
            technical_levels (dict): Support/resistance levels
            
        Returns:
            dict: Optimized stop-loss recommendation
        """
        try:
            # Base stop-loss calculation using ATR/volatility
            if volatility:
                atr_stop = self._calculate_atr_stop_loss(entry_price, signal_direction, volatility)
            else:
                atr_stop = self._calculate_percentage_stop_loss(entry_price, signal_direction)
                
            # Technical stop-loss based on support/resistance
            technical_stop = self._calculate_technical_stop_loss(
                entry_price, signal_direction, technical_levels
            )
            
            # Risk-based stop-loss
            risk_stop = self._calculate_risk_based_stop_loss(entry_price, signal_direction)
            
            # Choose optimal stop-loss
            optimal_stop = self._select_optimal_stop_loss(
                entry_price, signal_direction, atr_stop, technical_stop, risk_stop
            )
            
            # Calculate stop-loss metrics
            stop_distance = abs(entry_price - optimal_stop)
            stop_percentage = stop_distance / entry_price
            
            stop_loss_analysis = {
                'optimal_stop_loss': round(optimal_stop, 2),
                'stop_distance': round(stop_distance, 2),
                'stop_percentage': round(stop_percentage * 100, 2),
                'atr_stop': round(atr_stop, 2),
                'technical_stop': round(technical_stop, 2) if technical_stop else None,
                'risk_stop': round(risk_stop, 2),
                'stop_type': self._determine_stop_type(optimal_stop, atr_stop, technical_stop, risk_stop),
                'trailing_stop_recommendation': self._get_trailing_stop_recommendation(volatility)
            }
            
            return stop_loss_analysis
            
        except Exception as e:
            print(f"❌ Stop-loss optimization error: {e}")
            return self._get_default_stop_loss(entry_price, signal_direction)
            
    def calculate_take_profit(self, entry_price, stop_loss, signal_direction, confidence, risk_reward_target=2.0):
        """
        Calculate optimal take-profit levels
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop-loss price
            signal_direction (str): BUY or SELL
            confidence (float): Model confidence
            risk_reward_target (float): Target risk/reward ratio
            
        Returns:
            dict: Take-profit analysis
        """
        try:
            # Calculate risk amount
            risk_amount = abs(entry_price - stop_loss)
            
            # Adjust risk/reward target based on confidence
            adjusted_rr_target = self._adjust_risk_reward_target(risk_reward_target, confidence)
            
            # Calculate primary take-profit
            if signal_direction.upper() in ['BUY', 'STRONG_BUY']:
                primary_tp = entry_price + (risk_amount * adjusted_rr_target)
            else:
                primary_tp = entry_price - (risk_amount * adjusted_rr_target)
                
            # Calculate multiple take-profit levels
            tp_levels = self._calculate_multiple_tp_levels(
                entry_price, risk_amount, signal_direction, confidence
            )
            
            # Calculate probability of reaching targets
            tp_probabilities = self._calculate_tp_probabilities(confidence, tp_levels)
            
            take_profit_analysis = {
                'primary_take_profit': round(primary_tp, 2),
                'risk_reward_ratio': round(adjusted_rr_target, 1),
                'profit_amount': round(risk_amount * adjusted_rr_target, 2),
                'profit_percentage': round((risk_amount * adjusted_rr_target) / entry_price * 100, 2),
                'multiple_targets': tp_levels,
                'target_probabilities': tp_probabilities,
                'partial_profit_strategy': self._get_partial_profit_strategy(tp_levels),
                'trailing_profit_recommendation': self._get_trailing_profit_recommendation(confidence)
            }
            
            return take_profit_analysis
            
        except Exception as e:
            print(f"❌ Take-profit calculation error: {e}")
            return self._get_default_take_profit(entry_price, stop_loss, signal_direction)
            
    def assess_trade_risk(self, trade_params):
        """
        Comprehensive trade risk assessment
        
        Args:
            trade_params (dict): Trade parameters including entry, stop, size, etc.
            
        Returns:
            dict: Risk assessment results
        """
        try:
            entry_price = trade_params.get('entry_price', 0)
            stop_loss = trade_params.get('stop_loss', 0)
            position_size = trade_params.get('position_size', 0)
            confidence = trade_params.get('confidence', 50)
            
            # Calculate trade risk metrics
            risk_amount = abs(entry_price - stop_loss) * position_size
            risk_percentage = risk_amount / self.portfolio_value
            position_value = entry_price * position_size
            
            # Portfolio impact analysis
            portfolio_impact = self._analyze_portfolio_impact(position_value, risk_amount)
            
            # Risk-adjusted return expectation
            expected_return = self._calculate_expected_return(trade_params, confidence)
            
            # Correlation risk with existing positions
            correlation_risk = self._assess_correlation_risk(trade_params)
            
            # Market condition risk
            market_risk = self._assess_market_condition_risk()
            
            # Overall risk score (0-100, lower is better)
            overall_risk_score = self._calculate_overall_risk_score(
                risk_percentage, confidence, portfolio_impact, correlation_risk, market_risk
            )
            
            # Risk recommendation
            risk_recommendation = self._get_risk_recommendation(overall_risk_score, risk_percentage)
            
            risk_assessment = {
                'overall_risk_score': round(overall_risk_score, 1),
                'risk_level': self._categorize_risk_level(overall_risk_score),
                'risk_amount': round(risk_amount, 2),
                'risk_percentage': round(risk_percentage * 100, 2),
                'portfolio_impact': portfolio_impact,
                'expected_return': expected_return,
                'correlation_risk': correlation_risk,
                'market_risk': market_risk,
                'risk_recommendation': risk_recommendation,
                'max_acceptable_size': self._calculate_max_acceptable_size(trade_params),
                'risk_mitigation_suggestions': self._get_risk_mitigation_suggestions(overall_risk_score)
            }
            
            return risk_assessment
            
        except Exception as e:
            print(f"❌ Trade risk assessment error: {e}")
            return self._get_default_risk_assessment()
            
    def monitor_portfolio_risk(self):
        """
        Monitor overall portfolio risk and drawdown
        
        Returns:
            dict: Portfolio risk metrics
        """
        try:
            # Calculate current drawdown
            current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            
            # Update peak if necessary
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
                current_drawdown = 0.0
                
            self.current_drawdown = current_drawdown
            
            # Calculate portfolio risk metrics
            total_position_value = sum(pos.get('value', 0) for pos in self.current_positions)
            total_risk_amount = sum(pos.get('risk_amount', 0) for pos in self.current_positions)
            
            portfolio_utilization = total_position_value / self.portfolio_value
            portfolio_risk_percentage = total_risk_amount / self.portfolio_value
            
            # Risk alerts
            risk_alerts = []
            if current_drawdown > self.max_drawdown * 0.8:
                risk_alerts.append(f"Approaching maximum drawdown limit ({current_drawdown:.1%})")
            if portfolio_risk_percentage > self.max_portfolio_risk * 0.9:
                risk_alerts.append(f"High portfolio risk exposure ({portfolio_risk_percentage:.1%})")
            if portfolio_utilization > 0.8:
                risk_alerts.append(f"High portfolio utilization ({portfolio_utilization:.1%})")
                
            portfolio_metrics = {
                'portfolio_value': round(self.portfolio_value, 2),
                'peak_value': round(self.peak_portfolio_value, 2),
                'current_drawdown': round(current_drawdown * 100, 2),
                'max_drawdown_limit': round(self.max_drawdown * 100, 2),
                'total_position_value': round(total_position_value, 2),
                'portfolio_utilization': round(portfolio_utilization * 100, 2),
                'total_risk_amount': round(total_risk_amount, 2),
                'portfolio_risk_percentage': round(portfolio_risk_percentage * 100, 2),
                'max_portfolio_risk_limit': round(self.max_portfolio_risk * 100, 2),
                'active_positions': len(self.current_positions),
                'risk_alerts': risk_alerts,
                'risk_capacity_remaining': round((self.max_portfolio_risk - portfolio_risk_percentage) * 100, 2),
                'drawdown_capacity_remaining': round((self.max_drawdown - current_drawdown) * 100, 2)
            }
            
            return portfolio_metrics
            
        except Exception as e:
            print(f"❌ Portfolio risk monitoring error: {e}")
            return self._get_default_portfolio_metrics()
            
    # Helper methods
    def _calculate_confidence_factor(self, confidence):
        """Calculate position size adjustment based on confidence"""
        # Scale confidence from 0-100 to factor 0.5-1.5
        normalized_confidence = confidence / 100
        confidence_factor = 0.5 + normalized_confidence
        return min(1.5, max(0.5, confidence_factor))
        
    def _calculate_volatility_factor(self, volatility):
        """Calculate position size adjustment based on volatility"""
        if volatility is None:
            return 1.0
            
        # Higher volatility = smaller position
        if volatility > 0.03:  # High volatility
            return 0.7
        elif volatility > 0.02:  # Medium volatility
            return 0.85
        else:  # Low volatility
            return 1.0
            
    def _calculate_portfolio_risk_factor(self):
        """Calculate adjustment based on current portfolio risk"""
        current_risk = self.total_risk_exposure / self.portfolio_value
        risk_utilization = current_risk / self.max_portfolio_risk
        
        if risk_utilization > 0.8:
            return 0.5  # Significantly reduce new positions
        elif risk_utilization > 0.6:
            return 0.7
        else:
            return 1.0
            
    def _calculate_drawdown_factor(self):
        """Calculate adjustment based on current drawdown"""
        drawdown_utilization = self.current_drawdown / self.max_drawdown
        
        if drawdown_utilization > 0.8:
            return 0.3  # Drastically reduce positions near max drawdown
        elif drawdown_utilization > 0.6:
            return 0.6
        elif drawdown_utilization > 0.4:
            return 0.8
        else:
            return 1.0
            
    def _calculate_atr_stop_loss(self, entry_price, direction, volatility):
        """Calculate ATR-based stop loss"""
        atr_multiplier = 2.0  # Standard ATR multiplier
        atr_distance = volatility * atr_multiplier
        
        if direction.upper() in ['BUY', 'STRONG_BUY']:
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance
            
    def _calculate_percentage_stop_loss(self, entry_price, direction, percentage=0.02):
        """Calculate percentage-based stop loss"""
        if direction.upper() in ['BUY', 'STRONG_BUY']:
            return entry_price * (1 - percentage)
        else:
            return entry_price * (1 + percentage)
            
    def _calculate_technical_stop_loss(self, entry_price, direction, technical_levels):
        """Calculate technical stop loss based on support/resistance"""
        if not technical_levels:
            return None
            
        if direction.upper() in ['BUY', 'STRONG_BUY']:
            # Use support level for long positions
            support = technical_levels.get('support_level')
            if support and support < entry_price:
                return support * 0.999  # Slightly below support
        else:
            # Use resistance level for short positions
            resistance = technical_levels.get('resistance_level')
            if resistance and resistance > entry_price:
                return resistance * 1.001  # Slightly above resistance
                
        return None
        
    def _calculate_risk_based_stop_loss(self, entry_price, direction):
        """Calculate stop loss based on maximum acceptable risk"""
        max_risk_per_trade = 0.02  # 2% maximum risk
        
        if direction.upper() in ['BUY', 'STRONG_BUY']:
            return entry_price * (1 - max_risk_per_trade)
        else:
            return entry_price * (1 + max_risk_per_trade)
            
    def _select_optimal_stop_loss(self, entry_price, direction, atr_stop, technical_stop, risk_stop):
        """Select the optimal stop loss from calculated options"""
        stops = [atr_stop, risk_stop]
        if technical_stop:
            stops.append(technical_stop)
            
        if direction.upper() in ['BUY', 'STRONG_BUY']:
            # For long positions, use the highest (closest) stop
            return max(stops)
        else:
            # For short positions, use the lowest (closest) stop
            return min(stops)
            
    def _determine_stop_type(self, optimal_stop, atr_stop, technical_stop, risk_stop):
        """Determine which type of stop was selected"""
        if abs(optimal_stop - atr_stop) < 0.01:
            return 'ATR_BASED'
        elif technical_stop and abs(optimal_stop - technical_stop) < 0.01:
            return 'TECHNICAL'
        elif abs(optimal_stop - risk_stop) < 0.01:
            return 'RISK_BASED'
        else:
            return 'HYBRID'
            
    def _get_trailing_stop_recommendation(self, volatility):
        """Get trailing stop recommendation"""
        if volatility and volatility > 0.025:
            return "Use wide trailing stop due to high volatility"
        else:
            return "Standard trailing stop recommended"
            
    def _adjust_risk_reward_target(self, base_target, confidence):
        """Adjust risk/reward target based on confidence"""
        confidence_factor = confidence / 100
        # Higher confidence allows for higher targets
        adjusted_target = base_target * (0.8 + 0.4 * confidence_factor)
        return max(1.5, min(3.0, adjusted_target))
        
    def _calculate_multiple_tp_levels(self, entry_price, risk_amount, direction, confidence):
        """Calculate multiple take-profit levels"""
        base_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        # Adjust multipliers based on confidence
        confidence_factor = confidence / 100
        adjusted_multipliers = [m * (0.8 + 0.4 * confidence_factor) for m in base_multipliers]
        
        tp_levels = {}
        for i, multiplier in enumerate(adjusted_multipliers):
            if direction.upper() in ['BUY', 'STRONG_BUY']:
                tp_price = entry_price + (risk_amount * multiplier)
            else:
                tp_price = entry_price - (risk_amount * multiplier)
                
            tp_levels[f'TP{i+1}'] = round(tp_price, 2)
            
        return tp_levels
        
    def _calculate_tp_probabilities(self, confidence, tp_levels):
        """Calculate probability of reaching each TP level"""
        base_probability = confidence / 100
        probabilities = {}
        
        for i, (level_name, price) in enumerate(tp_levels.items()):
            # Probability decreases for higher targets
            prob = base_probability * (0.9 ** i)
            probabilities[level_name] = round(prob * 100, 1)
            
        return probabilities
        
    def _get_partial_profit_strategy(self, tp_levels):
        """Get partial profit taking strategy"""
        if len(tp_levels) >= 3:
            return "Take 30% at TP1, 40% at TP2, 30% at TP3+"
        else:
            return "Take 50% at TP1, 50% at TP2"
            
    def _get_trailing_profit_recommendation(self, confidence):
        """Get trailing profit recommendation"""
        if confidence > 80:
            return "Use aggressive trailing profit to maximize gains"
        elif confidence > 60:
            return "Use moderate trailing profit"
        else:
            return "Consider fixed targets due to lower confidence"
            
    def _analyze_portfolio_impact(self, position_value, risk_amount):
        """Analyze impact of trade on portfolio"""
        position_percentage = position_value / self.portfolio_value
        risk_percentage = risk_amount / self.portfolio_value
        
        return {
            'position_percentage': round(position_percentage * 100, 2),
            'risk_percentage': round(risk_percentage * 100, 2),
            'impact_level': 'HIGH' if position_percentage > 0.15 else 'MEDIUM' if position_percentage > 0.08 else 'LOW'
        }
        
    def _calculate_expected_return(self, trade_params, confidence):
        """Calculate expected return for the trade"""
        entry_price = trade_params.get('entry_price', 0)
        take_profit = trade_params.get('take_profit', 0)
        stop_loss = trade_params.get('stop_loss', 0)
        
        if not all([entry_price, take_profit, stop_loss]):
            return {'expected_return': 0, 'win_probability': 50}
            
        profit_amount = abs(take_profit - entry_price)
        loss_amount = abs(entry_price - stop_loss)
        
        # Win probability based on confidence
        win_probability = confidence / 100
        
        expected_return = (win_probability * profit_amount) - ((1 - win_probability) * loss_amount)
        expected_return_percentage = expected_return / entry_price
        
        return {
            'expected_return': round(expected_return, 2),
            'expected_return_percentage': round(expected_return_percentage * 100, 2),
            'win_probability': round(win_probability * 100, 1)
        }
        
    def _assess_correlation_risk(self, trade_params):
        """Assess correlation risk with existing positions"""
        # Simplified correlation assessment
        # In production, would analyze actual correlations
        
        if len(self.current_positions) == 0:
            return {'correlation_risk': 'LOW', 'diversification_score': 100}
            
        # Assume moderate correlation for gold positions
        correlation_risk = 'MEDIUM' if len(self.current_positions) > 2 else 'LOW'
        diversification_score = max(20, 100 - (len(self.current_positions) * 15))
        
        return {
            'correlation_risk': correlation_risk,
            'diversification_score': diversification_score,
            'existing_positions': len(self.current_positions)
        }
        
    def _assess_market_condition_risk(self):
        """Assess current market condition risk"""
        # Simplified market risk assessment
        # In production, would analyze volatility, trends, etc.
        
        return {
            'market_risk_level': 'MODERATE',
            'volatility_regime': 'NORMAL',
            'trend_strength': 'MODERATE',
            'market_sentiment': 'NEUTRAL'
        }
        
    def _calculate_overall_risk_score(self, risk_percentage, confidence, portfolio_impact, correlation_risk, market_risk):
        """Calculate overall risk score (0-100, lower is better)"""
        base_score = 50
        
        # Adjust for risk percentage
        if risk_percentage > 0.03:
            base_score += 30
        elif risk_percentage > 0.02:
            base_score += 15
        elif risk_percentage < 0.01:
            base_score -= 10
            
        # Adjust for confidence
        confidence_adjustment = (100 - confidence) / 4
        base_score += confidence_adjustment
        
        # Adjust for portfolio impact
        impact_level = portfolio_impact.get('impact_level', 'MEDIUM')
        if impact_level == 'HIGH':
            base_score += 20
        elif impact_level == 'LOW':
            base_score -= 10
            
        # Adjust for correlation risk
        corr_risk = correlation_risk.get('correlation_risk', 'MEDIUM')
        if corr_risk == 'HIGH':
            base_score += 15
        elif corr_risk == 'LOW':
            base_score -= 5
            
        return max(0, min(100, base_score))
        
    def _categorize_risk_level(self, risk_score):
        """Categorize risk level based on score"""
        if risk_score >= 80:
            return 'VERY_HIGH'
        elif risk_score >= 60:
            return 'HIGH'
        elif risk_score >= 40:
            return 'MEDIUM'
        elif risk_score >= 20:
            return 'LOW'
        else:
            return 'VERY_LOW'
            
    def _get_risk_recommendation(self, risk_score, risk_percentage):
        """Get risk-based recommendation"""
        if risk_score >= 80:
            return "HIGH RISK - Consider reducing position size or avoiding trade"
        elif risk_score >= 60:
            return "ELEVATED RISK - Proceed with caution, consider smaller position"
        elif risk_score >= 40:
            return "MODERATE RISK - Acceptable risk level with proper management"
        elif risk_score >= 20:
            return "LOW RISK - Good risk/reward setup"
        else:
            return "VERY LOW RISK - Excellent risk/reward opportunity"
            
    def _calculate_max_acceptable_size(self, trade_params):
        """Calculate maximum acceptable position size"""
        entry_price = trade_params.get('entry_price', 0)
        stop_loss = trade_params.get('stop_loss', 0)
        
        if not entry_price or not stop_loss:
            return 0
            
        risk_per_unit = abs(entry_price - stop_loss)
        max_risk_amount = self.portfolio_value * self.max_risk_per_trade
        max_size = max_risk_amount / risk_per_unit
        
        return round(max_size, 2)
        
    def _get_risk_mitigation_suggestions(self, risk_score):
        """Get risk mitigation suggestions"""
        suggestions = []
        
        if risk_score >= 60:
            suggestions.append("Reduce position size")
            suggestions.append("Tighten stop-loss")
            suggestions.append("Consider partial position entry")
            
        if risk_score >= 40:
            suggestions.append("Monitor position closely")
            suggestions.append("Set alerts for key levels")
            
        if not suggestions:
            suggestions.append("Current risk level is acceptable")
            
        return suggestions
        
    # Default return methods
    def _get_default_position_size(self):
        """Default position size in case of errors"""
        return {
            'position_size': 0.1,
            'position_value': 200.0,
            'risk_amount': 20.0,
            'risk_percentage': 0.02,
            'recommendation': 'Conservative position due to calculation error'
        }
        
    def _get_default_stop_loss(self, entry_price, direction):
        """Default stop loss in case of errors"""
        if direction.upper() in ['BUY', 'STRONG_BUY']:
            stop_loss = entry_price * 0.98
        else:
            stop_loss = entry_price * 1.02
            
        return {
            'optimal_stop_loss': round(stop_loss, 2),
            'stop_distance': round(abs(entry_price - stop_loss), 2),
            'stop_percentage': 2.0,
            'stop_type': 'DEFAULT'
        }
        
    def _get_default_take_profit(self, entry_price, stop_loss, direction):
        """Default take profit in case of errors"""
        risk_amount = abs(entry_price - stop_loss)
        
        if direction.upper() in ['BUY', 'STRONG_BUY']:
            take_profit = entry_price + (risk_amount * 2.0)
        else:
            take_profit = entry_price - (risk_amount * 2.0)
            
        return {
            'primary_take_profit': round(take_profit, 2),
            'risk_reward_ratio': 2.0,
            'profit_amount': round(risk_amount * 2.0, 2)
        }
        
    def _get_default_risk_assessment(self):
        """Default risk assessment in case of errors"""
        return {
            'overall_risk_score': 50,
            'risk_level': 'MEDIUM',
            'risk_recommendation': 'Standard risk management applies',
            'risk_mitigation_suggestions': ['Monitor position closely']
        }
        
    def _get_default_portfolio_metrics(self):
        """Default portfolio metrics in case of errors"""
        return {
            'portfolio_value': self.portfolio_value,
            'current_drawdown': 0.0,
            'portfolio_risk_percentage': 0.0,
            'risk_alerts': [],
            'active_positions': 0
        }
