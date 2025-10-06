"""Correlation Engine for real-time correlation calculations and monitoring.

This module provides comprehensive correlation analysis capabilities including:
- Real-time correlation calculations between assets
- Cross-asset correlation monitoring (stocks, bonds, commodities, crypto)
- Rolling correlation analysis and trend detection
- Correlation breakdown alerts and regime detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


class CorrelationRegime(Enum):
    """Enum for correlation regime classification."""
    HIGH_POSITIVE = "high_positive"      # > 0.7
    MODERATE_POSITIVE = "moderate_positive"  # 0.3 to 0.7
    LOW_CORRELATION = "low_correlation"   # -0.3 to 0.3
    MODERATE_NEGATIVE = "moderate_negative"  # -0.7 to -0.3
    HIGH_NEGATIVE = "high_negative"       # < -0.7


class AssetClass(Enum):
    """Enum for asset class classification."""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    REIT = "reit"
    ALTERNATIVE = "alternative"


class CorrelationAlert(Enum):
    """Enum for correlation alert types."""
    BREAKDOWN = "breakdown"       # Correlation dropped significantly
    SPIKE = "spike"              # Correlation increased significantly
    REGIME_CHANGE = "regime_change"  # Correlation regime changed
    DIVERGENCE = "divergence"     # Expected vs actual correlation divergence


@dataclass
class CorrelationPair:
    """Data class for correlation pair information."""
    asset1: str
    asset2: str
    correlation: float
    confidence: float
    regime: CorrelationRegime
    observations: int
    last_updated: datetime


@dataclass
class CorrelationBreakdown:
    """Data class for correlation breakdown events."""
    pair: str
    previous_correlation: float
    current_correlation: float
    change_magnitude: float
    breakdown_date: datetime
    alert_level: str  # high, medium, low


class RealTimeCorrelationEngine:
    """Engine for real-time correlation calculations and monitoring."""
    
    def __init__(self, window_size: int = 252, min_observations: int = 30):
        """Initialize the correlation engine.
        
        Args:
            window_size: Rolling window size for correlation calculations (default: 252 days)
            min_observations: Minimum observations required for valid correlation
        """
        self.window_size = window_size
        self.min_observations = min_observations
        
        # Asset classification mapping
        self.asset_classes = {
            # Equities
            'SPY': AssetClass.EQUITY, 'QQQ': AssetClass.EQUITY, 'IWM': AssetClass.EQUITY,
            'VTI': AssetClass.EQUITY, 'VOO': AssetClass.EQUITY, 'VXF': AssetClass.EQUITY,
            'EFA': AssetClass.EQUITY, 'EEM': AssetClass.EQUITY, 'VEA': AssetClass.EQUITY,
            
            # Sector ETFs
            'XLK': AssetClass.EQUITY, 'XLV': AssetClass.EQUITY, 'XLF': AssetClass.EQUITY,
            'XLY': AssetClass.EQUITY, 'XLP': AssetClass.EQUITY, 'XLE': AssetClass.EQUITY,
            'XLI': AssetClass.EQUITY, 'XLB': AssetClass.EQUITY, 'XLRE': AssetClass.REIT,
            'XLU': AssetClass.EQUITY, 'XLC': AssetClass.EQUITY,
            
            # Bonds
            'TLT': AssetClass.BOND, 'IEF': AssetClass.BOND, 'SHY': AssetClass.BOND,
            'LQD': AssetClass.BOND, 'HYG': AssetClass.BOND, 'JNK': AssetClass.BOND,
            'AGG': AssetClass.BOND, 'BND': AssetClass.BOND, 'VTEB': AssetClass.BOND,
            
            # Commodities
            'GLD': AssetClass.COMMODITY, 'SLV': AssetClass.COMMODITY, 'USO': AssetClass.COMMODITY,
            'UNG': AssetClass.COMMODITY, 'DBA': AssetClass.COMMODITY, 'DBC': AssetClass.COMMODITY,
            'PDBC': AssetClass.COMMODITY, 'GSG': AssetClass.COMMODITY,
            
            # Currencies
            'UUP': AssetClass.CURRENCY, 'FXE': AssetClass.CURRENCY, 'FXY': AssetClass.CURRENCY,
            'FXB': AssetClass.CURRENCY, 'CYB': AssetClass.CURRENCY,
            
            # Crypto
            'BTC-USD': AssetClass.CRYPTO, 'ETH-USD': AssetClass.CRYPTO, 'GBTC': AssetClass.CRYPTO,
            'ETHE': AssetClass.CRYPTO, 'BITW': AssetClass.CRYPTO,
            
            # REITs
            'VNQ': AssetClass.REIT, 'SCHH': AssetClass.REIT, 'RWR': AssetClass.REIT,
        }
        
        # Correlation thresholds for regime classification
        self.correlation_thresholds = {
            CorrelationRegime.HIGH_POSITIVE: 0.7,
            CorrelationRegime.MODERATE_POSITIVE: 0.3,
            CorrelationRegime.LOW_CORRELATION: -0.3,
            CorrelationRegime.MODERATE_NEGATIVE: -0.7
        }
        
        # Historical correlations for comparison
        self.correlation_history = {}
        self.correlation_alerts = []
    
    def calculate_rolling_correlations(self, price_data: Dict[str, pd.DataFrame], 
                                     return_type: str = 'close') -> Dict[str, Any]:
        """Calculate rolling correlations between all asset pairs.
        
        Args:
            price_data: Dictionary of asset symbol -> OHLCV DataFrame
            return_type: Type of return to use ('close', 'log', 'simple')
            
        Returns:
            Dictionary containing correlation matrix and analysis
        """
        try:
            if len(price_data) < 2:
                return {"error": "Need at least 2 assets for correlation analysis"}
            
            # Prepare returns data
            returns_data = self._prepare_returns_data(price_data, return_type)
            
            if returns_data.empty:
                return {"error": "No valid returns data available"}
            
            correlation_analysis = {
                "timestamp": datetime.now().isoformat(),
                "analysis_window": self.window_size,
                "num_assets": len(returns_data.columns),
                "correlation_matrix": {},
                "correlation_pairs": [],
                "regime_analysis": {},
                "correlation_statistics": {},
                "rolling_correlations": {},
                "alerts": []
            }
            
            # Calculate current correlation matrix
            current_correlations = returns_data.corr()
            correlation_analysis["correlation_matrix"] = current_correlations.to_dict()
            
            # Generate correlation pairs
            correlation_analysis["correlation_pairs"] = self._generate_correlation_pairs(
                current_correlations, returns_data
            )
            
            # Analyze correlation regimes
            correlation_analysis["regime_analysis"] = self._analyze_correlation_regimes(
                current_correlations
            )
            
            # Calculate correlation statistics
            correlation_analysis["correlation_statistics"] = self._calculate_correlation_statistics(
                current_correlations
            )
            
            # Calculate rolling correlations for time series analysis
            if len(returns_data) > self.window_size:
                correlation_analysis["rolling_correlations"] = self._calculate_rolling_correlation_series(
                    returns_data
                )
            
            # Detect correlation alerts
            correlation_analysis["alerts"] = self._detect_correlation_alerts(
                current_correlations, returns_data
            )
            
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlations: {e}")
            return {"error": str(e)}
    
    def monitor_cross_asset_correlations(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Monitor correlations across different asset classes.
        
        Args:
            price_data: Dictionary of asset symbol -> OHLCV DataFrame
            
        Returns:
            Dictionary containing cross-asset correlation analysis
        """
        try:
            cross_asset_analysis = {
                "timestamp": datetime.now().isoformat(),
                "asset_class_correlations": {},
                "diversification_analysis": {},
                "risk_on_risk_off": {},
                "flight_to_quality": {},
                "correlation_clusters": {},
                "portfolio_implications": {}
            }
            
            # Prepare returns data
            returns_data = self._prepare_returns_data(price_data, 'close')
            
            if returns_data.empty:
                return {"error": "No valid returns data for cross-asset analysis"}
            
            # Group assets by class
            asset_groups = self._group_assets_by_class(returns_data.columns)
            
            # Calculate inter-class correlations
            cross_asset_analysis["asset_class_correlations"] = self._calculate_inter_class_correlations(
                returns_data, asset_groups
            )
            
            # Analyze diversification benefits
            cross_asset_analysis["diversification_analysis"] = self._analyze_diversification_benefits(
                returns_data, asset_groups
            )
            
            # Detect risk-on/risk-off behavior
            cross_asset_analysis["risk_on_risk_off"] = self._analyze_risk_on_risk_off(
                returns_data, asset_groups
            )
            
            # Analyze flight-to-quality patterns
            cross_asset_analysis["flight_to_quality"] = self._analyze_flight_to_quality(
                returns_data, asset_groups
            )
            
            # Identify correlation clusters
            cross_asset_analysis["correlation_clusters"] = self._identify_correlation_clusters(
                returns_data
            )
            
            # Generate portfolio implications
            cross_asset_analysis["portfolio_implications"] = self._generate_portfolio_implications(
                cross_asset_analysis
            )
            
            return cross_asset_analysis
            
        except Exception as e:
            logger.error(f"Error monitoring cross-asset correlations: {e}")
            return {"error": str(e)}
    
    def detect_correlation_breakdowns(self, price_data: Dict[str, pd.DataFrame],
                                    lookback_periods: List[int] = [21, 63, 252]) -> Dict[str, Any]:
        """Detect correlation breakdowns and regime changes.
        
        Args:
            price_data: Dictionary of asset symbol -> OHLCV DataFrame
            lookback_periods: List of periods to analyze for breakdowns
            
        Returns:
            Dictionary containing breakdown analysis
        """
        try:
            breakdown_analysis = {
                "timestamp": datetime.now().isoformat(),
                "analysis_periods": lookback_periods,
                "breakdown_events": [],
                "regime_changes": [],
                "stability_metrics": {},
                "early_warning_signals": [],
                "breakdown_severity": {}
            }
            
            # Prepare returns data
            returns_data = self._prepare_returns_data(price_data, 'close')
            
            if len(returns_data) < max(lookback_periods) * 2:
                return {"error": "Insufficient data for breakdown analysis"}
            
            # Analyze each lookback period
            for period in lookback_periods:
                period_breakdowns = self._detect_period_breakdowns(returns_data, period)
                breakdown_analysis["breakdown_events"].extend(period_breakdowns)
            
            # Detect regime changes
            breakdown_analysis["regime_changes"] = self._detect_regime_changes(returns_data)
            
            # Calculate stability metrics
            breakdown_analysis["stability_metrics"] = self._calculate_stability_metrics(returns_data)
            
            # Generate early warning signals
            breakdown_analysis["early_warning_signals"] = self._generate_early_warning_signals(
                returns_data, breakdown_analysis["stability_metrics"]
            )
            
            # Assess breakdown severity
            breakdown_analysis["breakdown_severity"] = self._assess_breakdown_severity(
                breakdown_analysis["breakdown_events"]
            )
            
            return breakdown_analysis
            
        except Exception as e:
            logger.error(f"Error detecting correlation breakdowns: {e}")
            return {"error": str(e)}
    
    def _prepare_returns_data(self, price_data: Dict[str, pd.DataFrame], 
                            return_type: str = 'close') -> pd.DataFrame:
        """Prepare returns data from price data."""
        try:
            returns_dict = {}
            
            for symbol, data in price_data.items():
                if data.empty or 'Close' not in data.columns:
                    continue
                
                # Calculate returns based on type
                if return_type == 'log':
                    returns = np.log(data['Close'] / data['Close'].shift(1))
                else:  # simple returns
                    returns = data['Close'].pct_change(fill_method=None)
                
                # Remove NaN values and ensure sufficient data
                returns = returns.dropna()
                if len(returns) >= self.min_observations:
                    returns_dict[symbol] = returns
            
            if not returns_dict:
                return pd.DataFrame()
            
            # Align all return series to common date range
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()
            
            return returns_df
            
        except Exception as e:
            logger.error(f"Error preparing returns data: {e}")
            return pd.DataFrame()
    
    def _generate_correlation_pairs(self, correlation_matrix: pd.DataFrame,
                                  returns_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate detailed correlation pair analysis."""
        try:
            pairs = []
            symbols = correlation_matrix.index.tolist()
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j:  # Avoid duplicates and self-correlations
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        
                        # Classify correlation regime
                        regime = self._classify_correlation_regime(correlation)
                        
                        # Calculate confidence based on number of observations
                        observations = len(returns_data[[symbol1, symbol2]].dropna())
                        confidence = min(1.0, observations / self.window_size)
                        
                        # Get asset classes
                        class1 = self.asset_classes.get(symbol1, AssetClass.ALTERNATIVE)
                        class2 = self.asset_classes.get(symbol2, AssetClass.ALTERNATIVE)
                        
                        pair_info = {
                            "pair": f"{symbol1}-{symbol2}",
                            "asset1": symbol1,
                            "asset2": symbol2,
                            "asset_class1": class1.value,
                            "asset_class2": class2.value,
                            "correlation": float(correlation),
                            "regime": regime.value,
                            "confidence": float(confidence),
                            "observations": observations,
                            "cross_asset": class1 != class2,
                            "interpretation": self._interpret_correlation(correlation, class1, class2)
                        }
                        
                        pairs.append(pair_info)
            
            # Sort by absolute correlation (strongest first)
            pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return pairs
            
        except Exception as e:
            logger.error(f"Error generating correlation pairs: {e}")
            return []
    
    def _classify_correlation_regime(self, correlation: float) -> CorrelationRegime:
        """Classify correlation into regime based on thresholds."""
        if correlation > self.correlation_thresholds[CorrelationRegime.HIGH_POSITIVE]:
            return CorrelationRegime.HIGH_POSITIVE
        elif correlation > self.correlation_thresholds[CorrelationRegime.MODERATE_POSITIVE]:
            return CorrelationRegime.MODERATE_POSITIVE
        elif correlation > self.correlation_thresholds[CorrelationRegime.LOW_CORRELATION]:
            return CorrelationRegime.LOW_CORRELATION
        elif correlation > self.correlation_thresholds[CorrelationRegime.MODERATE_NEGATIVE]:
            return CorrelationRegime.MODERATE_NEGATIVE
        else:
            return CorrelationRegime.HIGH_NEGATIVE
    
    def _interpret_correlation(self, correlation: float, class1: AssetClass, 
                             class2: AssetClass) -> str:
        """Generate human-readable interpretation of correlation."""
        try:
            abs_corr = abs(correlation)
            direction = "positive" if correlation > 0 else "negative"
            
            if abs_corr > 0.8:
                strength = "very strong"
            elif abs_corr > 0.6:
                strength = "strong"
            elif abs_corr > 0.4:
                strength = "moderate"
            elif abs_corr > 0.2:
                strength = "weak"
            else:
                strength = "very weak"
            
            # Add context based on asset classes
            if class1 == class2:
                context = f"within {class1.value} asset class"
            else:
                context = f"between {class1.value} and {class2.value}"
            
            return f"{strength} {direction} correlation {context}"
            
        except Exception as e:
            logger.error(f"Error interpreting correlation: {e}")
            return "correlation analysis error"
    
    def _analyze_correlation_regimes(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation regimes across the matrix."""
        try:
            # Extract upper triangle (unique pairs)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            correlations = correlation_matrix.where(mask).stack().dropna()
            
            regime_counts = {regime.value: 0 for regime in CorrelationRegime}
            
            for corr in correlations:
                regime = self._classify_correlation_regime(corr)
                regime_counts[regime.value] += 1
            
            total_pairs = len(correlations)
            
            regime_analysis = {
                "total_pairs": total_pairs,
                "regime_distribution": regime_counts,
                "regime_percentages": {
                    regime: (count / total_pairs * 100) if total_pairs > 0 else 0
                    for regime, count in regime_counts.items()
                },
                "dominant_regime": max(regime_counts, key=regime_counts.get),
                "correlation_summary": {
                    "mean": float(correlations.mean()),
                    "median": float(correlations.median()),
                    "std": float(correlations.std()),
                    "min": float(correlations.min()),
                    "max": float(correlations.max())
                },
                "market_stress_indicator": self._calculate_market_stress_indicator(correlations)
            }
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing correlation regimes: {e}")
            return {}
    
    def _calculate_market_stress_indicator(self, correlations: pd.Series) -> Dict[str, Any]:
        """Calculate market stress indicator based on correlation patterns."""
        try:
            # High correlations often indicate market stress
            high_corr_threshold = 0.7
            high_correlations = correlations[correlations > high_corr_threshold]
            
            stress_score = len(high_correlations) / len(correlations) if len(correlations) > 0 else 0
            
            if stress_score > 0.6:
                stress_level = "high"
                interpretation = "High correlation regime suggests market stress or crisis conditions"
            elif stress_score > 0.3:
                stress_level = "medium"
                interpretation = "Moderate correlation levels indicate some market tension"
            else:
                stress_level = "low"
                interpretation = "Low correlation regime suggests normal market conditions"
            
            return {
                "stress_score": float(stress_score),
                "stress_level": stress_level,
                "high_correlation_count": len(high_correlations),
                "total_pairs": len(correlations),
                "interpretation": interpretation
            }
            
        except Exception as e:
            logger.error(f"Error calculating market stress indicator: {e}")
            return {}
    
    def _calculate_correlation_statistics(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive correlation statistics."""
        try:
            # Extract unique correlations (upper triangle)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            correlations = correlation_matrix.where(mask).stack().dropna()
            
            if correlations.empty:
                return {}
            
            statistics = {
                "basic_stats": {
                    "count": len(correlations),
                    "mean": float(correlations.mean()),
                    "median": float(correlations.median()),
                    "std": float(correlations.std()),
                    "min": float(correlations.min()),
                    "max": float(correlations.max()),
                    "skewness": float(correlations.skew()),
                    "kurtosis": float(correlations.kurtosis())
                },
                "percentiles": {
                    "p10": float(correlations.quantile(0.1)),
                    "p25": float(correlations.quantile(0.25)),
                    "p75": float(correlations.quantile(0.75)),
                    "p90": float(correlations.quantile(0.9))
                },
                "extreme_correlations": {
                    "highest_positive": float(correlations.max()),
                    "highest_negative": float(correlations.min()),
                    "most_extreme": float(correlations.abs().max())
                },
                "correlation_clusters": {
                    "high_positive": len(correlations[correlations > 0.7]),
                    "moderate_positive": len(correlations[(correlations > 0.3) & (correlations <= 0.7)]),
                    "low_correlation": len(correlations[abs(correlations) <= 0.3]),
                    "moderate_negative": len(correlations[(correlations < -0.3) & (correlations >= -0.7)]),
                    "high_negative": len(correlations[correlations < -0.7])
                }
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating correlation statistics: {e}")
            return {}
    
    def _calculate_rolling_correlation_series(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate rolling correlation time series for key pairs."""
        try:
            rolling_correlations = {}
            symbols = returns_data.columns.tolist()
            
            # Calculate rolling correlations for selected important pairs
            important_pairs = self._select_important_pairs(symbols)
            
            for asset1, asset2 in important_pairs:
                if asset1 in returns_data.columns and asset2 in returns_data.columns:
                    # Calculate rolling correlation
                    rolling_corr = returns_data[asset1].rolling(
                        window=min(63, len(returns_data) // 4)  # Quarterly window or shorter
                    ).corr(returns_data[asset2])
                    
                    rolling_corr = rolling_corr.dropna()
                    
                    if not rolling_corr.empty:
                        pair_key = f"{asset1}-{asset2}"
                        rolling_correlations[pair_key] = {
                            "current": float(rolling_corr.iloc[-1]),
                            "mean": float(rolling_corr.mean()),
                            "std": float(rolling_corr.std()),
                            "min": float(rolling_corr.min()),
                            "max": float(rolling_corr.max()),
                            "trend": self._analyze_correlation_trend(rolling_corr),
                            "stability": float(1.0 - rolling_corr.std()),  # Higher is more stable
                            "recent_change": float(rolling_corr.iloc[-1] - rolling_corr.iloc[-21]) if len(rolling_corr) > 21 else 0.0
                        }
            
            return {
                "rolling_correlations": rolling_correlations,
                "analysis_window": min(63, len(returns_data) // 4),
                "num_pairs_analyzed": len(rolling_correlations)
            }
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlation series: {e}")
            return {}
    
    def _select_important_pairs(self, symbols: List[str]) -> List[Tuple[str, str]]:
        """Select important asset pairs for detailed analysis."""
        important_pairs = []
        
        # Key benchmark pairs
        benchmarks = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
        
        for benchmark in benchmarks:
            if benchmark in symbols:
                for symbol in symbols:
                    if symbol != benchmark and (benchmark, symbol) not in important_pairs:
                        important_pairs.append((benchmark, symbol))
        
        # Cross-asset class pairs
        equity_etfs = [s for s in symbols if self.asset_classes.get(s) == AssetClass.EQUITY]
        bond_etfs = [s for s in symbols if self.asset_classes.get(s) == AssetClass.BOND]
        commodity_etfs = [s for s in symbols if self.asset_classes.get(s) == AssetClass.COMMODITY]
        
        # Add some cross-class pairs
        for equity in equity_etfs[:3]:  # Top 3 equity ETFs
            for bond in bond_etfs[:2]:   # Top 2 bond ETFs
                important_pairs.append((equity, bond))
            for commodity in commodity_etfs[:2]:  # Top 2 commodity ETFs
                important_pairs.append((equity, commodity))
        
        return important_pairs[:20]  # Limit to top 20 pairs
    
    def _analyze_correlation_trend(self, rolling_correlation: pd.Series) -> str:
        """Analyze trend in rolling correlation."""
        try:
            if len(rolling_correlation) < 10:
                return "insufficient_data"
            
            # Simple trend analysis using linear regression slope
            x = np.arange(len(rolling_correlation))
            y = rolling_correlation.values
            
            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error analyzing correlation trend: {e}")
            return "analysis_error"
    
    def _detect_correlation_alerts(self, current_correlations: pd.DataFrame,
                                 returns_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect correlation alerts and anomalies."""
        try:
            alerts = []
            
            # Compare with historical correlations if available
            if hasattr(self, 'previous_correlations') and self.previous_correlations is not None:
                alerts.extend(self._detect_correlation_changes(
                    self.previous_correlations, current_correlations
                ))
            
            # Detect extreme correlations
            alerts.extend(self._detect_extreme_correlations(current_correlations))
            
            # Detect correlation spikes in rolling window
            if len(returns_data) > 63:  # Need sufficient data
                alerts.extend(self._detect_correlation_spikes(returns_data))
            
            # Store current correlations for next comparison
            self.previous_correlations = current_correlations.copy()
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error detecting correlation alerts: {e}")
            return []
    
    def _detect_correlation_changes(self, previous: pd.DataFrame, 
                                   current: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect significant changes in correlations."""
        alerts = []
        
        try:
            # Find common assets
            common_assets = set(previous.index) & set(current.index)
            
            for asset1 in common_assets:
                for asset2 in common_assets:
                    if asset1 >= asset2:  # Avoid duplicates
                        continue
                    
                    prev_corr = previous.loc[asset1, asset2]
                    curr_corr = current.loc[asset1, asset2]
                    
                    change = abs(curr_corr - prev_corr)
                    
                    # Alert thresholds
                    if change > 0.3:  # Major change
                        alert_level = "high"
                    elif change > 0.2:  # Moderate change
                        alert_level = "medium"
                    elif change > 0.1:  # Minor change
                        alert_level = "low"
                    else:
                        continue
                    
                    alert_type = CorrelationAlert.BREAKDOWN if curr_corr < prev_corr else CorrelationAlert.SPIKE
                    
                    alerts.append({
                        "type": alert_type.value,
                        "pair": f"{asset1}-{asset2}",
                        "previous_correlation": float(prev_corr),
                        "current_correlation": float(curr_corr),
                        "change": float(curr_corr - prev_corr),
                        "change_magnitude": float(change),
                        "alert_level": alert_level,
                        "timestamp": datetime.now().isoformat(),
                        "description": f"Correlation change of {change:.2f} detected between {asset1} and {asset2}"
                    })
            
        except Exception as e:
            logger.error(f"Error detecting correlation changes: {e}")
        
        return alerts
    
    def _detect_extreme_correlations(self, correlations: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect extreme correlation values."""
        alerts = []
        
        try:
            symbols = correlations.index.tolist()
            
            for i, asset1 in enumerate(symbols):
                for j, asset2 in enumerate(symbols):
                    if i >= j:  # Avoid duplicates and self-correlations
                        continue
                    
                    corr = correlations.loc[asset1, asset2]
                    
                    # Check for extreme correlations
                    if abs(corr) > 0.95:  # Very high correlation
                        class1 = self.asset_classes.get(asset1, AssetClass.ALTERNATIVE)
                        class2 = self.asset_classes.get(asset2, AssetClass.ALTERNATIVE)
                        
                        # Only alert if it's unexpected (different asset classes)
                        if class1 != class2:
                            alerts.append({
                                "type": "extreme_correlation",
                                "pair": f"{asset1}-{asset2}",
                                "correlation": float(corr),
                                "asset_class1": class1.value,
                                "asset_class2": class2.value,
                                "alert_level": "high",
                                "timestamp": datetime.now().isoformat(),
                                "description": f"Extreme correlation ({corr:.3f}) between different asset classes: {asset1} ({class1.value}) and {asset2} ({class2.value})"
                            })
            
        except Exception as e:
            logger.error(f"Error detecting extreme correlations: {e}")
        
        return alerts
    
    def _detect_correlation_spikes(self, returns_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect correlation spikes in rolling windows."""
        alerts = []
        
        try:
            # Calculate short vs long term correlations
            short_window = 21  # 1 month
            long_window = 63   # 3 months
            
            if len(returns_data) < long_window:
                return alerts
            
            symbols = returns_data.columns.tolist()
            recent_data = returns_data.tail(short_window)
            longer_data = returns_data.tail(long_window)
            
            for i, asset1 in enumerate(symbols):
                for j, asset2 in enumerate(symbols):
                    if i >= j:
                        continue
                    
                    # Calculate short and long term correlations
                    short_corr = recent_data[asset1].corr(recent_data[asset2])
                    long_corr = longer_data[asset1].corr(longer_data[asset2])
                    
                    if pd.isna(short_corr) or pd.isna(long_corr):
                        continue
                    
                    change = abs(short_corr - long_corr)
                    
                    if change > 0.4:  # Significant spike
                        alerts.append({
                            "type": "correlation_spike",
                            "pair": f"{asset1}-{asset2}",
                            "short_term_correlation": float(short_corr),
                            "long_term_correlation": float(long_corr),
                            "change": float(short_corr - long_corr),
                            "change_magnitude": float(change),
                            "alert_level": "high" if change > 0.6 else "medium",
                            "timestamp": datetime.now().isoformat(),
                            "description": f"Correlation spike detected: {asset1}-{asset2} correlation changed from {long_corr:.2f} to {short_corr:.2f}"
                        })
            
        except Exception as e:
            logger.error(f"Error detecting correlation spikes: {e}")
        
        return alerts
    
    def _group_assets_by_class(self, symbols: List[str]) -> Dict[AssetClass, List[str]]:
        """Group assets by their asset class."""
        groups = defaultdict(list)
        
        for symbol in symbols:
            asset_class = self.asset_classes.get(symbol, AssetClass.ALTERNATIVE)
            groups[asset_class].append(symbol)
        
        return dict(groups)
    
    def _calculate_inter_class_correlations(self, returns_data: pd.DataFrame,
                                          asset_groups: Dict[AssetClass, List[str]]) -> Dict[str, Any]:
        """Calculate correlations between asset classes."""
        try:
            inter_class_correlations = {}
            
            # Calculate average returns for each asset class
            class_returns = {}
            for asset_class, symbols in asset_groups.items():
                if len(symbols) > 0:
                    available_symbols = [s for s in symbols if s in returns_data.columns]
                    if available_symbols:
                        # Equal-weighted average of all assets in the class
                        class_returns[asset_class.value] = returns_data[available_symbols].mean(axis=1)
            
            # Calculate correlations between asset classes
            if len(class_returns) > 1:
                class_returns_df = pd.DataFrame(class_returns)
                correlation_matrix = class_returns_df.corr()
                inter_class_correlations = correlation_matrix.to_dict()
            
            return {
                "correlation_matrix": inter_class_correlations,
                "asset_classes_analyzed": list(class_returns.keys()),
                "interpretation": self._interpret_inter_class_correlations(inter_class_correlations)
            }
            
        except Exception as e:
            logger.error(f"Error calculating inter-class correlations: {e}")
            return {}
    
    def _interpret_inter_class_correlations(self, correlations: Dict[str, Dict[str, float]]) -> Dict[str, str]:
        """Interpret inter-class correlation patterns."""
        interpretations = {}
        
        try:
            for class1, corr_dict in correlations.items():
                for class2, corr_value in corr_dict.items():
                    if class1 != class2:
                        pair = f"{class1}-{class2}"
                        
                        if corr_value > 0.5:
                            interpretations[pair] = f"Strong positive correlation ({corr_value:.2f}) suggests risk-on behavior"
                        elif corr_value < -0.3:
                            interpretations[pair] = f"Negative correlation ({corr_value:.2f}) provides diversification benefits"
                        else:
                            interpretations[pair] = f"Moderate correlation ({corr_value:.2f}) indicates partial diversification"
            
        except Exception as e:
            logger.error(f"Error interpreting inter-class correlations: {e}")
        
        return interpretations
    
    def _analyze_diversification_benefits(self, returns_data: pd.DataFrame,
                                        asset_groups: Dict[AssetClass, List[str]]) -> Dict[str, Any]:
        """Analyze diversification benefits across asset classes."""
        try:
            diversification_analysis = {
                "portfolio_correlation": 0.0,
                "diversification_ratio": 1.0,
                "effective_number_of_assets": 1.0,
                "concentration_risk": {},
                "diversification_score": 0.0
            }
            
            if len(returns_data.columns) < 2:
                return diversification_analysis
            
            # Calculate equal-weighted portfolio correlation
            correlation_matrix = returns_data.corr()
            n_assets = len(correlation_matrix)
            
            # Average correlation (excluding diagonal)
            mask = ~np.eye(n_assets, dtype=bool)
            avg_correlation = correlation_matrix.values[mask].mean()
            diversification_analysis["portfolio_correlation"] = float(avg_correlation)
            
            # Diversification ratio
            individual_vol = returns_data.std()
            portfolio_returns = returns_data.mean(axis=1)
            portfolio_vol = portfolio_returns.std()
            weighted_avg_vol = individual_vol.mean()
            
            if portfolio_vol > 0:
                diversification_ratio = weighted_avg_vol / portfolio_vol
                diversification_analysis["diversification_ratio"] = float(diversification_ratio)
            
            # Effective number of assets (based on correlation)
            effective_n = 1 + (n_assets - 1) * (1 - avg_correlation)
            diversification_analysis["effective_number_of_assets"] = float(effective_n)
            
            # Concentration risk by asset class
            total_assets = len(returns_data.columns)
            for asset_class, symbols in asset_groups.items():
                class_weight = len(symbols) / total_assets
                diversification_analysis["concentration_risk"][asset_class.value] = float(class_weight)
            
            # Overall diversification score (0-100)
            correlation_score = max(0, (1 - avg_correlation) * 100)
            class_diversity_score = (1 - max(diversification_analysis["concentration_risk"].values())) * 100
            diversification_score = (correlation_score + class_diversity_score) / 2
            diversification_analysis["diversification_score"] = float(diversification_score)
            
            return diversification_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing diversification benefits: {e}")
            return {}
    
    def _analyze_risk_on_risk_off(self, returns_data: pd.DataFrame,
                                asset_groups: Dict[AssetClass, List[str]]) -> Dict[str, Any]:
        """Analyze risk-on/risk-off market behavior."""
        try:
            risk_analysis = {
                "current_regime": "neutral",
                "risk_indicators": {},
                "regime_strength": 0.5,
                "regime_signals": []
            }
            
            # Define risk-on and risk-off assets
            risk_on_assets = []
            risk_off_assets = []
            
            # Risk-on: Equities, especially growth and small-cap
            if AssetClass.EQUITY in asset_groups:
                equity_assets = asset_groups[AssetClass.EQUITY]
                growth_proxies = [s for s in equity_assets if s in ['QQQ', 'XLK', 'IWM']]
                risk_on_assets.extend(growth_proxies)
            
            # Risk-off: Bonds, especially treasuries
            if AssetClass.BOND in asset_groups:
                bond_assets = asset_groups[AssetClass.BOND]
                safe_haven_bonds = [s for s in bond_assets if s in ['TLT', 'IEF', 'SHY']]
                risk_off_assets.extend(safe_haven_bonds)
            
            # Calculate recent performance
            if len(returns_data) >= 21:  # Need at least 1 month of data
                recent_returns = returns_data.tail(21)  # Last month
                
                risk_on_performance = 0.0
                risk_off_performance = 0.0
                
                if risk_on_assets:
                    available_risk_on = [s for s in risk_on_assets if s in recent_returns.columns]
                    if available_risk_on:
                        risk_on_performance = recent_returns[available_risk_on].mean().mean()
                
                if risk_off_assets:
                    available_risk_off = [s for s in risk_off_assets if s in recent_returns.columns]
                    if available_risk_off:
                        risk_off_performance = recent_returns[available_risk_off].mean().mean()
                
                # Determine regime
                performance_diff = risk_on_performance - risk_off_performance
                
                if performance_diff > 0.01:  # 1% threshold
                    risk_analysis["current_regime"] = "risk_on"
                    risk_analysis["regime_strength"] = min(1.0, abs(performance_diff) * 50)
                elif performance_diff < -0.01:
                    risk_analysis["current_regime"] = "risk_off"
                    risk_analysis["regime_strength"] = min(1.0, abs(performance_diff) * 50)
                else:
                    risk_analysis["current_regime"] = "neutral"
                    risk_analysis["regime_strength"] = 0.5
                
                risk_analysis["risk_indicators"] = {
                    "risk_on_performance": float(risk_on_performance),
                    "risk_off_performance": float(risk_off_performance),
                    "performance_differential": float(performance_diff),
                    "risk_on_assets_analyzed": len([s for s in risk_on_assets if s in recent_returns.columns]),
                    "risk_off_assets_analyzed": len([s for s in risk_off_assets if s in recent_returns.columns])
                }
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing risk-on/risk-off behavior: {e}")
            return {}
    
    def _analyze_flight_to_quality(self, returns_data: pd.DataFrame,
                                 asset_groups: Dict[AssetClass, List[str]]) -> Dict[str, Any]:
        """Analyze flight-to-quality patterns."""
        try:
            flight_analysis = {
                "flight_detected": False,
                "flight_intensity": 0.0,
                "safe_haven_performance": {},
                "risky_asset_performance": {},
                "flight_indicators": []
            }
            
            if len(returns_data) < 21:
                return flight_analysis
            
            recent_returns = returns_data.tail(21)  # Last month
            
            # Define safe haven assets
            safe_haven_assets = []
            if AssetClass.BOND in asset_groups:
                treasury_bonds = [s for s in asset_groups[AssetClass.BOND] if s in ['TLT', 'IEF']]
                safe_haven_assets.extend(treasury_bonds)
            
            if AssetClass.COMMODITY in asset_groups:
                gold_assets = [s for s in asset_groups[AssetClass.COMMODITY] if s in ['GLD']]
                safe_haven_assets.extend(gold_assets)
            
            # Define risky assets
            risky_assets = []
            if AssetClass.EQUITY in asset_groups:
                equity_assets = asset_groups[AssetClass.EQUITY][:5]  # Top 5 equity assets
                risky_assets.extend(equity_assets)
            
            # Calculate performance metrics
            if safe_haven_assets and risky_assets:
                available_safe = [s for s in safe_haven_assets if s in recent_returns.columns]
                available_risky = [s for s in risky_assets if s in recent_returns.columns]
                
                if available_safe and available_risky:
                    safe_perf = recent_returns[available_safe].mean().mean()
                    risky_perf = recent_returns[available_risky].mean().mean()
                    
                    # Flight to quality occurs when safe havens outperform significantly
                    performance_gap = safe_perf - risky_perf
                    
                    flight_analysis["safe_haven_performance"] = {
                        "assets": available_safe,
                        "average_return": float(safe_perf),
                        "volatility": float(recent_returns[available_safe].mean().std())
                    }
                    
                    flight_analysis["risky_asset_performance"] = {
                        "assets": available_risky,
                        "average_return": float(risky_perf),
                        "volatility": float(recent_returns[available_risky].mean().std())
                    }
                    
                    # Detect flight to quality
                    if performance_gap > 0.005:  # 0.5% threshold
                        flight_analysis["flight_detected"] = True
                        flight_analysis["flight_intensity"] = min(1.0, performance_gap * 100)
                        flight_analysis["flight_indicators"].append(
                            f"Safe haven assets outperformed by {performance_gap:.2%}"
                        )
                    
                    # Check correlation patterns
                    equity_bond_corr = 0.0
                    if available_risky and available_safe:
                        for risky in available_risky[:2]:  # Check top 2
                            for safe in available_safe[:2]:
                                corr = recent_returns[risky].corr(recent_returns[safe])
                                if not pd.isna(corr):
                                    equity_bond_corr += corr
                        
                        equity_bond_corr /= min(4, len(available_risky) * len(available_safe))
                        
                        if equity_bond_corr < -0.3:  # Strong negative correlation
                            flight_analysis["flight_indicators"].append(
                                f"Strong negative correlation ({equity_bond_corr:.2f}) between risky and safe assets"
                            )
            
            return flight_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing flight to quality: {e}")
            return {}
    
    def _identify_correlation_clusters(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify clusters of highly correlated assets."""
        try:
            clustering_analysis = {
                "clusters": [],
                "cluster_summary": {},
                "outlier_assets": [],
                "cluster_methodology": "hierarchical"
            }
            
            if len(returns_data.columns) < 3:
                return clustering_analysis
            
            correlation_matrix = returns_data.corr()
            
            # Simple clustering based on correlation thresholds
            high_corr_threshold = 0.7
            clusters = []
            assigned_assets = set()
            
            symbols = correlation_matrix.index.tolist()
            
            for i, asset1 in enumerate(symbols):
                if asset1 in assigned_assets:
                    continue
                
                cluster = [asset1]
                assigned_assets.add(asset1)
                
                for j, asset2 in enumerate(symbols):
                    if asset2 != asset1 and asset2 not in assigned_assets:
                        corr = correlation_matrix.loc[asset1, asset2]
                        if corr > high_corr_threshold:
                            cluster.append(asset2)
                            assigned_assets.add(asset2)
                
                if len(cluster) > 1:  # Only include clusters with multiple assets
                    cluster_info = {
                        "cluster_id": len(clusters) + 1,
                        "assets": cluster,
                        "size": len(cluster),
                        "average_correlation": self._calculate_cluster_avg_correlation(
                            cluster, correlation_matrix
                        ),
                        "dominant_asset_class": self._get_dominant_asset_class(cluster)
                    }
                    clusters.append(cluster_info)
            
            # Identify outlier assets (low correlation with others)
            outliers = []
            for asset in symbols:
                if asset not in assigned_assets:
                    avg_corr = correlation_matrix[asset].drop(asset).abs().mean()
                    if avg_corr < 0.3:  # Low average correlation
                        outliers.append({
                            "asset": asset,
                            "average_correlation": float(avg_corr),
                            "asset_class": self.asset_classes.get(asset, AssetClass.ALTERNATIVE).value
                        })
            
            clustering_analysis["clusters"] = clusters
            clustering_analysis["outlier_assets"] = outliers
            clustering_analysis["cluster_summary"] = {
                "total_clusters": len(clusters),
                "total_outliers": len(outliers),
                "largest_cluster_size": max([c["size"] for c in clusters]) if clusters else 0,
                "average_cluster_size": np.mean([c["size"] for c in clusters]) if clusters else 0
            }
            
            return clustering_analysis
            
        except Exception as e:
            logger.error(f"Error identifying correlation clusters: {e}")
            return {}
    
    def _calculate_cluster_avg_correlation(self, cluster: List[str], 
                                         correlation_matrix: pd.DataFrame) -> float:
        """Calculate average correlation within a cluster."""
        try:
            if len(cluster) < 2:
                return 0.0
            
            correlations = []
            for i, asset1 in enumerate(cluster):
                for j, asset2 in enumerate(cluster):
                    if i < j:  # Avoid duplicates and self-correlations
                        corr = correlation_matrix.loc[asset1, asset2]
                        correlations.append(corr)
            
            return float(np.mean(correlations)) if correlations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating cluster average correlation: {e}")
            return 0.0
    
    def _get_dominant_asset_class(self, cluster: List[str]) -> str:
        """Get the dominant asset class in a cluster."""
        try:
            class_counts = defaultdict(int)
            
            for asset in cluster:
                asset_class = self.asset_classes.get(asset, AssetClass.ALTERNATIVE)
                class_counts[asset_class.value] += 1
            
            return max(class_counts, key=class_counts.get) if class_counts else "unknown"
            
        except Exception as e:
            logger.error(f"Error getting dominant asset class: {e}")
            return "unknown"
    
    def _generate_portfolio_implications(self, cross_asset_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portfolio implications from cross-asset analysis."""
        try:
            implications = {
                "diversification_recommendations": [],
                "risk_management_insights": [],
                "allocation_suggestions": {},
                "correlation_hedging": [],
                "regime_based_positioning": {}
            }
            
            # Diversification recommendations
            diversification = cross_asset_analysis.get("diversification_analysis", {})
            if diversification:
                div_score = diversification.get("diversification_score", 0)
                
                if div_score < 50:
                    implications["diversification_recommendations"].append(
                        "Portfolio shows high concentration risk - consider adding assets from different classes"
                    )
                elif div_score > 80:
                    implications["diversification_recommendations"].append(
                        "Portfolio is well-diversified across asset classes"
                    )
                
                # Check concentration risk
                concentration = diversification.get("concentration_risk", {})
                max_concentration = max(concentration.values()) if concentration else 0
                
                if max_concentration > 0.6:
                    dominant_class = max(concentration, key=concentration.get)
                    implications["diversification_recommendations"].append(
                        f"High concentration in {dominant_class} ({max_concentration:.1%}) - consider rebalancing"
                    )
            
            # Risk management insights
            risk_regime = cross_asset_analysis.get("risk_on_risk_off", {})
            current_regime = risk_regime.get("current_regime", "neutral")
            
            if current_regime == "risk_off":
                implications["risk_management_insights"].append(
                    "Risk-off environment detected - consider defensive positioning"
                )
                implications["regime_based_positioning"]["defensive"] = [
                    "Increase allocation to treasury bonds",
                    "Consider defensive sectors (utilities, consumer staples)",
                    "Reduce exposure to high-beta assets"
                ]
            elif current_regime == "risk_on":
                implications["risk_management_insights"].append(
                    "Risk-on environment - growth assets may outperform"
                )
                implications["regime_based_positioning"]["aggressive"] = [
                    "Consider increasing equity allocation",
                    "Growth and small-cap stocks may outperform",
                    "Reduce defensive positions"
                ]
            
            # Flight to quality implications
            flight_analysis = cross_asset_analysis.get("flight_to_quality", {})
            if flight_analysis.get("flight_detected", False):
                implications["risk_management_insights"].append(
                    "Flight to quality detected - safe haven assets outperforming"
                )
                implications["correlation_hedging"].append({
                    "strategy": "safe_haven_allocation",
                    "description": "Increase allocation to treasury bonds and gold",
                    "target_allocation": "10-20%"
                })
            
            return implications
            
        except Exception as e:
            logger.error(f"Error generating portfolio implications: {e}")
            return {}
    
    def _detect_period_breakdowns(self, returns_data: pd.DataFrame, 
                                period: int) -> List[Dict[str, Any]]:
        """Detect correlation breakdowns for a specific period."""
        breakdowns = []
        
        try:
            if len(returns_data) < period * 2:
                return breakdowns
            
            # Calculate correlations for current and previous periods
            current_period = returns_data.tail(period)
            previous_period = returns_data.iloc[-period*2:-period]
            
            current_corr = current_period.corr()
            previous_corr = previous_period.corr()
            
            # Find significant changes
            symbols = current_corr.index.tolist()
            
            for i, asset1 in enumerate(symbols):
                for j, asset2 in enumerate(symbols):
                    if i >= j:
                        continue
                    
                    curr = current_corr.loc[asset1, asset2]
                    prev = previous_corr.loc[asset1, asset2]
                    
                    if pd.isna(curr) or pd.isna(prev):
                        continue
                    
                    change = curr - prev
                    
                    if abs(change) > 0.3:  # Significant breakdown/spike
                        breakdown = CorrelationBreakdown(
                            pair=f"{asset1}-{asset2}",
                            previous_correlation=float(prev),
                            current_correlation=float(curr),
                            change_magnitude=float(abs(change)),
                            breakdown_date=datetime.now(),
                            alert_level="high" if abs(change) > 0.5 else "medium"
                        )
                        
                        breakdowns.append({
                            "pair": breakdown.pair,
                            "previous_correlation": breakdown.previous_correlation,
                            "current_correlation": breakdown.current_correlation,
                            "change": float(change),
                            "change_magnitude": breakdown.change_magnitude,
                            "period_days": period,
                            "alert_level": breakdown.alert_level,
                            "breakdown_type": "correlation_drop" if change < 0 else "correlation_spike",
                            "timestamp": breakdown.breakdown_date.isoformat()
                        })
            
        except Exception as e:
            logger.error(f"Error detecting period breakdowns: {e}")
        
        return breakdowns
    
    def _detect_regime_changes(self, returns_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect correlation regime changes."""
        regime_changes = []
        
        try:
            if len(returns_data) < 126:  # Need at least 6 months
                return regime_changes
            
            # Calculate correlations for different periods
            periods = [21, 63, 126]  # 1 month, 3 months, 6 months
            regime_data = {}
            
            for period in periods:
                if len(returns_data) >= period:
                    period_data = returns_data.tail(period)
                    corr_matrix = period_data.corr()
                    
                    # Extract unique correlations
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    correlations = corr_matrix.where(mask).stack().dropna()
                    
                    regime_data[period] = {
                        "mean_correlation": float(correlations.mean()),
                        "correlation_std": float(correlations.std()),
                        "high_corr_count": len(correlations[correlations > 0.7]),
                        "total_pairs": len(correlations)
                    }
            
            # Compare regimes across periods
            if len(regime_data) >= 2:
                short_regime = regime_data[21]  # 1 month
                long_regime = regime_data[126]  # 6 months
                
                mean_change = abs(short_regime["mean_correlation"] - long_regime["mean_correlation"])
                
                if mean_change > 0.2:  # Significant regime change
                    regime_changes.append({
                        "change_type": "correlation_regime_shift",
                        "short_term_mean": short_regime["mean_correlation"],
                        "long_term_mean": long_regime["mean_correlation"],
                        "change_magnitude": float(mean_change),
                        "direction": "increasing" if short_regime["mean_correlation"] > long_regime["mean_correlation"] else "decreasing",
                        "alert_level": "high" if mean_change > 0.3 else "medium",
                        "timestamp": datetime.now().isoformat(),
                        "description": f"Correlation regime shift detected: mean correlation changed by {mean_change:.2f}"
                    })
            
        except Exception as e:
            logger.error(f"Error detecting regime changes: {e}")
        
        return regime_changes
    
    def _calculate_stability_metrics(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation stability metrics."""
        try:
            stability_metrics = {
                "correlation_volatility": 0.0,
                "regime_persistence": 0.0,
                "breakdown_frequency": 0.0,
                "stability_score": 0.5
            }
            
            if len(returns_data) < 63:
                return stability_metrics
            
            # Calculate rolling correlation volatility
            window = 21
            symbols = returns_data.columns.tolist()
            
            if len(symbols) >= 2:
                # Use first two assets as proxy
                asset1, asset2 = symbols[0], symbols[1]
                rolling_corr = returns_data[asset1].rolling(window).corr(returns_data[asset2])
                rolling_corr = rolling_corr.dropna()
                
                if len(rolling_corr) > 10:
                    corr_volatility = rolling_corr.std()
                    stability_metrics["correlation_volatility"] = float(corr_volatility)
                    
                    # Lower volatility = higher stability
                    stability_score = max(0, 1 - corr_volatility * 2)
                    stability_metrics["stability_score"] = float(stability_score)
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Error calculating stability metrics: {e}")
            return {}
    
    def _generate_early_warning_signals(self, returns_data: pd.DataFrame,
                                       stability_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate early warning signals for correlation breakdowns."""
        warnings = []
        
        try:
            # Check correlation volatility
            corr_vol = stability_metrics.get("correlation_volatility", 0)
            
            if corr_vol > 0.3:  # High correlation volatility
                warnings.append({
                    "signal_type": "high_correlation_volatility",
                    "severity": "medium",
                    "metric_value": corr_vol,
                    "description": f"High correlation volatility ({corr_vol:.2f}) may indicate upcoming breakdown",
                    "recommended_action": "Monitor correlation patterns closely and consider defensive positioning"
                })
            
            # Check for increasing volatility trend
            if len(returns_data) >= 63:
                recent_vol = returns_data.tail(21).std().mean()
                longer_vol = returns_data.tail(63).std().mean()
                
                vol_change = (recent_vol - longer_vol) / longer_vol if longer_vol > 0 else 0
                
                if vol_change > 0.5:  # 50% increase in volatility
                    warnings.append({
                        "signal_type": "volatility_spike",
                        "severity": "high",
                        "metric_value": float(vol_change),
                        "description": f"Volatility increased by {vol_change:.1%}, correlation breakdown may follow",
                        "recommended_action": "Prepare for potential correlation changes and review hedging strategies"
                    })
            
        except Exception as e:
            logger.error(f"Error generating early warning signals: {e}")
        
        return warnings
    
    def _assess_breakdown_severity(self, breakdown_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the overall severity of correlation breakdowns."""
        try:
            severity_assessment = {
                "total_breakdowns": len(breakdown_events),
                "severity_distribution": {"high": 0, "medium": 0, "low": 0},
                "average_magnitude": 0.0,
                "systemic_risk_level": "low",
                "most_affected_pairs": []
            }
            
            if not breakdown_events:
                return severity_assessment
            
            # Count by severity
            for event in breakdown_events:
                severity = event.get("alert_level", "low")
                severity_assessment["severity_distribution"][severity] += 1
            
            # Calculate average magnitude
            magnitudes = [event.get("change_magnitude", 0) for event in breakdown_events]
            severity_assessment["average_magnitude"] = float(np.mean(magnitudes))
            
            # Assess systemic risk
            high_severity_count = severity_assessment["severity_distribution"]["high"]
            total_count = len(breakdown_events)
            
            if high_severity_count > 5 or (total_count > 0 and high_severity_count / total_count > 0.3):
                severity_assessment["systemic_risk_level"] = "high"
            elif high_severity_count > 2 or (total_count > 0 and high_severity_count / total_count > 0.1):
                severity_assessment["systemic_risk_level"] = "medium"
            
            # Identify most affected pairs
            severity_assessment["most_affected_pairs"] = sorted(
                breakdown_events,
                key=lambda x: x.get("change_magnitude", 0),
                reverse=True
            )[:5]  # Top 5 most affected pairs
            
            return severity_assessment
            
        except Exception as e:
            logger.error(f"Error assessing breakdown severity: {e}")
            return {}