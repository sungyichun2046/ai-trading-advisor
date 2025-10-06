"""Sector Analysis Engine for ETF performance analysis and rotation signals.

This module provides comprehensive sector analysis capabilities including:
- ETF sector performance tracking and comparison
- Sector rotation signal generation
- Momentum ranking and relative strength analysis
- Sector allocation recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SectorStrength(Enum):
    """Enum for sector strength classification."""
    VERY_STRONG = "very_strong"
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    VERY_WEAK = "very_weak"


class RotationSignal(Enum):
    """Enum for sector rotation signals."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class SectorInfo:
    """Data class for sector information."""
    
    def __init__(self, symbol: str, name: str, sector: str, 
                 performance: float, momentum: float, 
                 relative_strength: float, strength: SectorStrength):
        self.symbol = symbol
        self.name = name
        self.sector = sector
        self.performance = performance
        self.momentum = momentum
        self.relative_strength = relative_strength
        self.strength = strength


class SectorPerformanceAnalyzer:
    """Analyzes ETF sector performance and generates insights."""
    
    def __init__(self):
        """Initialize the sector performance analyzer."""
        self.sector_etfs = {
            # Technology
            'XLK': 'Technology Select Sector SPDR Fund',
            'VGT': 'Vanguard Information Technology ETF',
            'FTEC': 'Fidelity MSCI Information Technology ETF',
            
            # Healthcare
            'XLV': 'Health Care Select Sector SPDR Fund',
            'VHT': 'Vanguard Health Care ETF',
            'FHLC': 'Fidelity MSCI Health Care ETF',
            
            # Financial
            'XLF': 'Financial Select Sector SPDR Fund',
            'VFH': 'Vanguard Financials ETF',
            'FNCL': 'Fidelity MSCI Financials ETF',
            
            # Consumer Discretionary
            'XLY': 'Consumer Discretionary Select Sector SPDR Fund',
            'VCR': 'Vanguard Consumer Discretionary ETF',
            'FDIS': 'Fidelity MSCI Consumer Discretionary ETF',
            
            # Consumer Staples
            'XLP': 'Consumer Staples Select Sector SPDR Fund',
            'VDC': 'Vanguard Consumer Staples ETF',
            'FSTA': 'Fidelity MSCI Consumer Staples ETF',
            
            # Energy
            'XLE': 'Energy Select Sector SPDR Fund',
            'VDE': 'Vanguard Energy ETF',
            'FENY': 'Fidelity MSCI Energy ETF',
            
            # Industrials
            'XLI': 'Industrial Select Sector SPDR Fund',
            'VIS': 'Vanguard Industrials ETF',
            'FIDU': 'Fidelity MSCI Industrials ETF',
            
            # Materials
            'XLB': 'Materials Select Sector SPDR Fund',
            'VAW': 'Vanguard Materials ETF',
            'FMAT': 'Fidelity MSCI Materials ETF',
            
            # Real Estate
            'XLRE': 'Real Estate Select Sector SPDR Fund',
            'VNQ': 'Vanguard Real Estate ETF',
            'FREL': 'Fidelity MSCI Real Estate ETF',
            
            # Utilities
            'XLU': 'Utilities Select Sector SPDR Fund',
            'VPU': 'Vanguard Utilities ETF',
            'FUTY': 'Fidelity MSCI Utilities ETF',
            
            # Communication Services
            'XLC': 'Communication Services Select Sector SPDR Fund',
            'VOX': 'Vanguard Communication Services ETF',
            'FCOM': 'Fidelity MSCI Communication Services ETF'
        }
        
        self.sector_mapping = {
            'XLK': 'Technology', 'VGT': 'Technology', 'FTEC': 'Technology',
            'XLV': 'Healthcare', 'VHT': 'Healthcare', 'FHLC': 'Healthcare',
            'XLF': 'Financial', 'VFH': 'Financial', 'FNCL': 'Financial',
            'XLY': 'Consumer Discretionary', 'VCR': 'Consumer Discretionary', 'FDIS': 'Consumer Discretionary',
            'XLP': 'Consumer Staples', 'VDC': 'Consumer Staples', 'FSTA': 'Consumer Staples',
            'XLE': 'Energy', 'VDE': 'Energy', 'FENY': 'Energy',
            'XLI': 'Industrials', 'VIS': 'Industrials', 'FIDU': 'Industrials',
            'XLB': 'Materials', 'VAW': 'Materials', 'FMAT': 'Materials',
            'XLRE': 'Real Estate', 'VNQ': 'Real Estate', 'FREL': 'Real Estate',
            'XLU': 'Utilities', 'VPU': 'Utilities', 'FUTY': 'Utilities',
            'XLC': 'Communication Services', 'VOX': 'Communication Services', 'FCOM': 'Communication Services'
        }
        
        # Performance calculation periods
        self.performance_periods = {
            'short_term': 5,      # 5 days
            'medium_term': 21,    # 21 days (1 month)
            'long_term': 63,      # 63 days (3 months)
            'quarterly': 126      # 126 days (6 months)
        }
    
    def calculate_sector_performance(self, price_data: Dict[str, pd.DataFrame], 
                                   benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate comprehensive sector performance metrics.
        
        Args:
            price_data: Dictionary of ETF symbol -> OHLCV DataFrame
            benchmark_data: Optional benchmark data (e.g., SPY) for relative performance
            
        Returns:
            Dictionary containing sector performance analysis
        """
        try:
            if not price_data:
                return {"error": "No price data provided"}
            
            analysis_results = {
                "timestamp": datetime.now().isoformat(),
                "sector_performance": {},
                "sector_rankings": {},
                "performance_summary": {},
                "rotation_candidates": {},
                "risk_metrics": {}
            }
            
            # Calculate performance for each ETF
            for symbol, data in price_data.items():
                if symbol not in self.sector_etfs:
                    continue
                    
                if data.empty or 'Close' not in data.columns:
                    logger.warning(f"Invalid data for {symbol}")
                    continue
                
                try:
                    sector_perf = self._calculate_etf_performance(symbol, data, benchmark_data)
                    analysis_results["sector_performance"][symbol] = sector_perf
                    
                except Exception as e:
                    logger.error(f"Error calculating performance for {symbol}: {e}")
                    analysis_results["sector_performance"][symbol] = {"error": str(e)}
            
            # Generate sector rankings
            analysis_results["sector_rankings"] = self._generate_sector_rankings(
                analysis_results["sector_performance"]
            )
            
            # Create performance summary
            analysis_results["performance_summary"] = self._create_performance_summary(
                analysis_results["sector_performance"]
            )
            
            # Identify rotation candidates
            analysis_results["rotation_candidates"] = self._identify_rotation_candidates(
                analysis_results["sector_performance"]
            )
            
            # Calculate risk metrics
            analysis_results["risk_metrics"] = self._calculate_risk_metrics(price_data)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error in sector performance analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_etf_performance(self, symbol: str, data: pd.DataFrame, 
                                  benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Calculate performance metrics for a single ETF."""
        try:
            performance = {
                "symbol": symbol,
                "sector": self.sector_mapping.get(symbol, "Unknown"),
                "name": self.sector_etfs.get(symbol, "Unknown ETF"),
                "current_price": float(data['Close'].iloc[-1]),
                "period_returns": {},
                "relative_performance": {},
                "momentum_indicators": {},
                "volatility_metrics": {},
                "trend_analysis": {}
            }
            
            # Calculate returns for different periods
            for period_name, days in self.performance_periods.items():
                if len(data) > days:
                    period_return = self._calculate_period_return(data, days)
                    performance["period_returns"][period_name] = period_return
                    
                    # Calculate relative performance vs benchmark if available
                    if benchmark_data is not None and not benchmark_data.empty:
                        rel_perf = self._calculate_relative_performance(
                            data, benchmark_data, days
                        )
                        performance["relative_performance"][period_name] = rel_perf
            
            # Calculate momentum indicators
            performance["momentum_indicators"] = self._calculate_momentum_indicators(data)
            
            # Calculate volatility metrics
            performance["volatility_metrics"] = self._calculate_volatility_metrics(data)
            
            # Perform trend analysis
            performance["trend_analysis"] = self._analyze_etf_trend(data)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating ETF performance for {symbol}: {e}")
            return {"error": str(e)}
    
    def _calculate_period_return(self, data: pd.DataFrame, days: int) -> float:
        """Calculate return over specified number of days."""
        if len(data) <= days:
            return 0.0
        
        start_price = data['Close'].iloc[-days-1]
        end_price = data['Close'].iloc[-1]
        return float((end_price - start_price) / start_price)
    
    def _calculate_relative_performance(self, etf_data: pd.DataFrame, 
                                      benchmark_data: pd.DataFrame, days: int) -> float:
        """Calculate relative performance vs benchmark."""
        try:
            if len(etf_data) <= days or len(benchmark_data) <= days:
                return 0.0
            
            etf_return = self._calculate_period_return(etf_data, days)
            benchmark_return = self._calculate_period_return(benchmark_data, days)
            
            return float(etf_return - benchmark_return)
            
        except Exception as e:
            logger.error(f"Error calculating relative performance: {e}")
            return 0.0
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators for the ETF."""
        try:
            if len(data) < 50:
                return {}
            
            close_prices = data['Close']
            
            # Rate of Change (ROC) for different periods
            roc_10 = float((close_prices.iloc[-1] - close_prices.iloc[-11]) / close_prices.iloc[-11])
            roc_20 = float((close_prices.iloc[-1] - close_prices.iloc[-21]) / close_prices.iloc[-21])
            
            # Price momentum (current price vs moving averages)
            sma_20 = close_prices.rolling(20).mean()
            sma_50 = close_prices.rolling(50).mean()
            
            price_vs_sma20 = float((close_prices.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1])
            price_vs_sma50 = float((close_prices.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1])
            
            # Moving average slope (trend strength)
            ma_slope_20 = float((sma_20.iloc[-1] - sma_20.iloc[-6]) / sma_20.iloc[-6])
            
            return {
                "roc_10_day": roc_10,
                "roc_20_day": roc_20,
                "price_vs_sma20": price_vs_sma20,
                "price_vs_sma50": price_vs_sma50,
                "ma_slope_20": ma_slope_20
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    def _calculate_volatility_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility metrics for the ETF."""
        try:
            if len(data) < 21:
                return {}
            
            close_prices = data['Close']
            returns = close_prices.pct_change(fill_method=None).dropna()
            
            # Standard volatility measures
            vol_20d_series = returns.rolling(20).std() * np.sqrt(252)
            vol_60d_series = returns.rolling(60).std() * np.sqrt(252)
            volatility_20d = float(vol_20d_series.iloc[-1]) if not vol_20d_series.empty else 0.0
            volatility_60d = float(vol_60d_series.iloc[-1]) if not vol_60d_series.empty else 0.0
            
            # Average True Range (ATR) based volatility
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr_series = true_range.rolling(14).mean()
            atr_14 = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
            atr_percentage = float(atr_14 / close_prices.iloc[-1]) if close_prices.iloc[-1] != 0 else 0.0
            
            return {
                "volatility_20d": volatility_20d,
                "volatility_60d": volatility_60d,
                "atr_14": atr_14,
                "atr_percentage": atr_percentage,
                "current_volatility_regime": "high" if volatility_20d > 0.25 else "medium" if volatility_20d > 0.15 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {}
    
    def _analyze_etf_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend characteristics of the ETF."""
        try:
            if len(data) < 50:
                return {}
            
            close_prices = data['Close']
            
            # Moving averages for trend identification
            sma_20 = close_prices.rolling(20).mean()
            sma_50 = close_prices.rolling(50).mean()
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            
            # Trend direction
            short_term_trend = "bullish" if sma_20.iloc[-1] > sma_20.iloc[-6] else "bearish"
            long_term_trend = "bullish" if sma_50.iloc[-1] > sma_50.iloc[-11] else "bearish"
            
            # MACD for trend momentum
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_histogram = macd - macd_signal
            
            # Trend strength (R-squared of linear regression)
            x = np.arange(len(close_prices.tail(20)))
            y = close_prices.tail(20).values
            correlation = np.corrcoef(x, y)[0, 1]
            trend_strength = float(correlation ** 2)
            
            return {
                "short_term_trend": short_term_trend,
                "long_term_trend": long_term_trend,
                "trend_strength": trend_strength,
                "macd_current": float(macd.iloc[-1]),
                "macd_signal": float(macd_signal.iloc[-1]),
                "macd_histogram": float(macd_histogram.iloc[-1]),
                "price_above_sma20": bool(close_prices.iloc[-1] > sma_20.iloc[-1]),
                "price_above_sma50": bool(close_prices.iloc[-1] > sma_50.iloc[-1]),
                "sma20_above_sma50": bool(sma_20.iloc[-1] > sma_50.iloc[-1])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ETF trend: {e}")
            return {}
    
    def _generate_sector_rankings(self, sector_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rankings of sectors by various metrics."""
        try:
            rankings = {
                "by_short_term_return": [],
                "by_medium_term_return": [],
                "by_long_term_return": [],
                "by_relative_strength": [],
                "by_momentum": [],
                "by_risk_adjusted_return": []
            }
            
            valid_sectors = []
            for symbol, perf in sector_performance.items():
                if "error" in perf or not isinstance(perf, dict):
                    continue
                valid_sectors.append((symbol, perf))
            
            if not valid_sectors:
                return rankings
            
            # Rank by short-term returns
            short_term_ranked = sorted(
                valid_sectors,
                key=lambda x: x[1].get("period_returns", {}).get("short_term", 0),
                reverse=True
            )
            rankings["by_short_term_return"] = [
                {
                    "symbol": symbol,
                    "sector": perf.get("sector", "Unknown"),
                    "return": perf.get("period_returns", {}).get("short_term", 0),
                    "rank": i + 1
                }
                for i, (symbol, perf) in enumerate(short_term_ranked)
            ]
            
            # Rank by medium-term returns
            medium_term_ranked = sorted(
                valid_sectors,
                key=lambda x: x[1].get("period_returns", {}).get("medium_term", 0),
                reverse=True
            )
            rankings["by_medium_term_return"] = [
                {
                    "symbol": symbol,
                    "sector": perf.get("sector", "Unknown"),
                    "return": perf.get("period_returns", {}).get("medium_term", 0),
                    "rank": i + 1
                }
                for i, (symbol, perf) in enumerate(medium_term_ranked)
            ]
            
            # Rank by momentum (ROC 20-day)
            momentum_ranked = sorted(
                valid_sectors,
                key=lambda x: x[1].get("momentum_indicators", {}).get("roc_20_day", 0),
                reverse=True
            )
            rankings["by_momentum"] = [
                {
                    "symbol": symbol,
                    "sector": perf.get("sector", "Unknown"),
                    "momentum": perf.get("momentum_indicators", {}).get("roc_20_day", 0),
                    "rank": i + 1
                }
                for i, (symbol, perf) in enumerate(momentum_ranked)
            ]
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error generating sector rankings: {e}")
            return {}
    
    def _create_performance_summary(self, sector_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of overall sector performance."""
        try:
            summary = {
                "total_sectors_analyzed": 0,
                "best_performing_sector": {},
                "worst_performing_sector": {},
                "average_performance": {},
                "performance_dispersion": {},
                "sector_breadth": {}
            }
            
            valid_performances = []
            for symbol, perf in sector_performance.items():
                if "error" not in perf and isinstance(perf, dict):
                    valid_performances.append((symbol, perf))
            
            if not valid_performances:
                return summary
            
            summary["total_sectors_analyzed"] = len(valid_performances)
            
            # Find best and worst performers
            medium_term_returns = [
                (symbol, perf.get("period_returns", {}).get("medium_term", 0))
                for symbol, perf in valid_performances
            ]
            
            if medium_term_returns:
                best_symbol, best_return = max(medium_term_returns, key=lambda x: x[1])
                worst_symbol, worst_return = min(medium_term_returns, key=lambda x: x[1])
                
                summary["best_performing_sector"] = {
                    "symbol": best_symbol,
                    "sector": sector_performance[best_symbol].get("sector", "Unknown"),
                    "return": best_return
                }
                
                summary["worst_performing_sector"] = {
                    "symbol": worst_symbol,
                    "sector": sector_performance[worst_symbol].get("sector", "Unknown"),
                    "return": worst_return
                }
                
                # Calculate average performance
                returns = [ret for _, ret in medium_term_returns]
                summary["average_performance"] = {
                    "mean_return": float(np.mean(returns)),
                    "median_return": float(np.median(returns)),
                    "std_return": float(np.std(returns))
                }
                
                # Calculate performance dispersion
                summary["performance_dispersion"] = {
                    "range": float(best_return - worst_return),
                    "coefficient_of_variation": float(np.std(returns) / np.mean(returns)) if np.mean(returns) != 0 else 0
                }
                
                # Calculate sector breadth (how many sectors are outperforming)
                positive_returns = len([ret for ret in returns if ret > 0])
                summary["sector_breadth"] = {
                    "positive_sectors": positive_returns,
                    "negative_sectors": len(returns) - positive_returns,
                    "breadth_ratio": float(positive_returns / len(returns))
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating performance summary: {e}")
            return {}
    
    def _identify_rotation_candidates(self, sector_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Identify sectors for potential rotation strategies."""
        try:
            candidates = {
                "rotation_into": [],  # Strong sectors to buy
                "rotation_out_of": [],  # Weak sectors to sell
                "momentum_plays": [],  # High momentum sectors
                "value_opportunities": [],  # Underperforming but potentially recovering
                "rotation_signals": {}
            }
            
            for symbol, perf in sector_performance.items():
                if "error" in perf or not isinstance(perf, dict):
                    continue
                
                try:
                    # Extract key metrics
                    short_return = perf.get("period_returns", {}).get("short_term", 0)
                    medium_return = perf.get("period_returns", {}).get("medium_term", 0)
                    momentum = perf.get("momentum_indicators", {}).get("roc_20_day", 0)
                    rel_strength = perf.get("relative_performance", {}).get("medium_term", 0)
                    trend = perf.get("trend_analysis", {})
                    
                    # Generate rotation signal
                    rotation_signal = self._generate_rotation_signal(
                        short_return, medium_return, momentum, rel_strength, trend
                    )
                    
                    candidates["rotation_signals"][symbol] = {
                        "signal": rotation_signal.value,
                        "strength": self._calculate_signal_strength(
                            short_return, medium_return, momentum, rel_strength
                        ),
                        "sector": perf.get("sector", "Unknown"),
                        "rationale": self._generate_signal_rationale(
                            rotation_signal, short_return, medium_return, momentum, rel_strength
                        )
                    }
                    
                    # Categorize based on signals
                    if rotation_signal in [RotationSignal.STRONG_BUY, RotationSignal.BUY]:
                        candidates["rotation_into"].append({
                            "symbol": symbol,
                            "sector": perf.get("sector", "Unknown"),
                            "signal": rotation_signal.value,
                            "medium_return": medium_return,
                            "momentum": momentum
                        })
                    elif rotation_signal in [RotationSignal.STRONG_SELL, RotationSignal.SELL]:
                        candidates["rotation_out_of"].append({
                            "symbol": symbol,
                            "sector": perf.get("sector", "Unknown"),
                            "signal": rotation_signal.value,
                            "medium_return": medium_return,
                            "momentum": momentum
                        })
                    
                    # High momentum plays
                    if momentum > 0.1 and medium_return > 0.05:  # 10% momentum, 5% medium-term return
                        candidates["momentum_plays"].append({
                            "symbol": symbol,
                            "sector": perf.get("sector", "Unknown"),
                            "momentum": momentum,
                            "return": medium_return
                        })
                
                except Exception as e:
                    logger.error(f"Error processing rotation candidate {symbol}: {e}")
                    continue
            
            # Sort candidates by strength
            candidates["rotation_into"].sort(key=lambda x: x.get("momentum", 0), reverse=True)
            candidates["rotation_out_of"].sort(key=lambda x: x.get("momentum", 0))
            candidates["momentum_plays"].sort(key=lambda x: x.get("momentum", 0), reverse=True)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error identifying rotation candidates: {e}")
            return {}
    
    def _generate_rotation_signal(self, short_return: float, medium_return: float,
                                momentum: float, rel_strength: float, 
                                trend: Dict[str, Any]) -> RotationSignal:
        """Generate rotation signal based on multiple factors."""
        try:
            score = 0
            
            # Short-term performance scoring
            if short_return > 0.05:  # 5%
                score += 2
            elif short_return > 0.02:  # 2%
                score += 1
            elif short_return < -0.05:  # -5%
                score -= 2
            elif short_return < -0.02:  # -2%
                score -= 1
            
            # Medium-term performance scoring
            if medium_return > 0.1:  # 10%
                score += 2
            elif medium_return > 0.05:  # 5%
                score += 1
            elif medium_return < -0.1:  # -10%
                score -= 2
            elif medium_return < -0.05:  # -5%
                score -= 1
            
            # Momentum scoring
            if momentum > 0.08:  # 8%
                score += 2
            elif momentum > 0.04:  # 4%
                score += 1
            elif momentum < -0.08:  # -8%
                score -= 2
            elif momentum < -0.04:  # -4%
                score -= 1
            
            # Relative strength scoring
            if rel_strength > 0.05:  # 5% outperformance
                score += 1
            elif rel_strength < -0.05:  # 5% underperformance
                score -= 1
            
            # Trend analysis scoring
            if trend:
                if (trend.get("short_term_trend") == "bullish" and 
                    trend.get("long_term_trend") == "bullish"):
                    score += 1
                elif (trend.get("short_term_trend") == "bearish" and 
                      trend.get("long_term_trend") == "bearish"):
                    score -= 1
                
                if trend.get("price_above_sma20", False) and trend.get("price_above_sma50", False):
                    score += 1
                elif not trend.get("price_above_sma20", True) and not trend.get("price_above_sma50", True):
                    score -= 1
            
            # Convert score to signal
            if score >= 6:
                return RotationSignal.STRONG_BUY
            elif score >= 3:
                return RotationSignal.BUY
            elif score <= -6:
                return RotationSignal.STRONG_SELL
            elif score <= -3:
                return RotationSignal.SELL
            else:
                return RotationSignal.HOLD
                
        except Exception as e:
            logger.error(f"Error generating rotation signal: {e}")
            return RotationSignal.HOLD
    
    def _calculate_signal_strength(self, short_return: float, medium_return: float,
                                 momentum: float, rel_strength: float) -> float:
        """Calculate the strength of the rotation signal."""
        try:
            # Combine metrics with weights
            strength = (
                abs(short_return) * 0.2 +
                abs(medium_return) * 0.3 +
                abs(momentum) * 0.3 +
                abs(rel_strength) * 0.2
            )
            
            return min(float(strength), 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5
    
    def _generate_signal_rationale(self, signal: RotationSignal, short_return: float,
                                 medium_return: float, momentum: float, 
                                 rel_strength: float) -> str:
        """Generate human-readable rationale for the rotation signal."""
        try:
            rationale_parts = []
            
            if signal in [RotationSignal.STRONG_BUY, RotationSignal.BUY]:
                if medium_return > 0.05:
                    rationale_parts.append(f"strong medium-term performance ({medium_return:.1%})")
                if momentum > 0.05:
                    rationale_parts.append(f"positive momentum ({momentum:.1%})")
                if rel_strength > 0.02:
                    rationale_parts.append(f"outperforming market ({rel_strength:.1%})")
            
            elif signal in [RotationSignal.STRONG_SELL, RotationSignal.SELL]:
                if medium_return < -0.05:
                    rationale_parts.append(f"weak medium-term performance ({medium_return:.1%})")
                if momentum < -0.05:
                    rationale_parts.append(f"negative momentum ({momentum:.1%})")
                if rel_strength < -0.02:
                    rationale_parts.append(f"underperforming market ({rel_strength:.1%})")
            
            else:
                rationale_parts.append("mixed signals, neutral outlook")
            
            return "; ".join(rationale_parts) if rationale_parts else "insufficient data for clear signal"
            
        except Exception as e:
            logger.error(f"Error generating signal rationale: {e}")
            return "analysis error"
    
    def _calculate_risk_metrics(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate risk metrics across sectors."""
        try:
            risk_metrics = {
                "sector_correlations": {},
                "volatility_rankings": [],
                "risk_adjusted_performance": {},
                "diversification_benefits": {}
            }
            
            # Extract returns for correlation calculation
            returns_data = {}
            for symbol, data in price_data.items():
                if symbol in self.sector_etfs and not data.empty and 'Close' in data.columns:
                    returns = data['Close'].pct_change(fill_method=None).dropna()
                    if len(returns) > 20:  # Minimum data requirement
                        returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return risk_metrics
            
            # Calculate correlations
            symbols = list(returns_data.keys())
            correlation_matrix = {}
            
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i != j:
                        # Align the series and calculate correlation
                        aligned_data = pd.concat([
                            returns_data[symbol1], 
                            returns_data[symbol2]
                        ], axis=1, keys=[symbol1, symbol2]).dropna()
                        
                        if len(aligned_data) > 10:
                            corr = float(aligned_data[symbol1].corr(aligned_data[symbol2]))
                            correlation_matrix[symbol1][symbol2] = corr
                        else:
                            correlation_matrix[symbol1][symbol2] = 0.0
                    else:
                        correlation_matrix[symbol1][symbol2] = 1.0
            
            risk_metrics["sector_correlations"] = correlation_matrix
            
            # Calculate volatility rankings
            volatility_data = []
            for symbol, returns in returns_data.items():
                vol = float(returns.std() * np.sqrt(252))  # Annualized volatility
                volatility_data.append({
                    "symbol": symbol,
                    "sector": self.sector_mapping.get(symbol, "Unknown"),
                    "volatility": vol
                })
            
            risk_metrics["volatility_rankings"] = sorted(
                volatility_data, 
                key=lambda x: x["volatility"]
            )
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}


class SectorRotationEngine:
    """Engine for generating sector rotation strategies and signals."""
    
    def __init__(self):
        """Initialize the sector rotation engine."""
        self.performance_analyzer = SectorPerformanceAnalyzer()
        
        # Rotation strategy parameters
        self.rotation_params = {
            "momentum_threshold": 0.05,      # 5% momentum threshold
            "performance_threshold": 0.03,   # 3% performance threshold
            "correlation_threshold": 0.7,    # High correlation threshold
            "volatility_penalty": 0.5,       # Penalty for high volatility
            "rebalance_frequency": "monthly" # Rebalancing frequency
        }
    
    def generate_rotation_strategy(self, price_data: Dict[str, pd.DataFrame], 
                                 current_allocation: Optional[Dict[str, float]] = None,
                                 benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate comprehensive sector rotation strategy.
        
        Args:
            price_data: Dictionary of ETF symbol -> OHLCV DataFrame
            current_allocation: Current portfolio allocation by sector
            benchmark_data: Optional benchmark data for relative analysis
            
        Returns:
            Dictionary containing rotation strategy recommendations
        """
        try:
            strategy = {
                "timestamp": datetime.now().isoformat(),
                "analysis_period": "current",
                "sector_analysis": {},
                "rotation_recommendations": {},
                "portfolio_optimization": {},
                "risk_management": {},
                "execution_plan": {}
            }
            
            # Perform sector analysis
            sector_analysis = self.performance_analyzer.calculate_sector_performance(
                price_data, benchmark_data
            )
            strategy["sector_analysis"] = sector_analysis
            
            # Generate rotation recommendations
            strategy["rotation_recommendations"] = self._generate_rotation_recommendations(
                sector_analysis, current_allocation
            )
            
            # Optimize portfolio allocation
            strategy["portfolio_optimization"] = self._optimize_sector_allocation(
                sector_analysis, current_allocation
            )
            
            # Add risk management considerations
            strategy["risk_management"] = self._add_risk_management(
                sector_analysis, strategy["portfolio_optimization"]
            )
            
            # Create execution plan
            strategy["execution_plan"] = self._create_execution_plan(
                strategy["rotation_recommendations"], 
                strategy["portfolio_optimization"],
                current_allocation
            )
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating rotation strategy: {e}")
            return {"error": str(e)}
    
    def _generate_rotation_recommendations(self, sector_analysis: Dict[str, Any], 
                                         current_allocation: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Generate specific rotation recommendations."""
        try:
            recommendations = {
                "top_sectors_to_buy": [],
                "sectors_to_reduce": [],
                "sectors_to_avoid": [],
                "momentum_opportunities": [],
                "contrarian_plays": [],
                "overall_market_sentiment": "neutral"
            }
            
            rotation_candidates = sector_analysis.get("rotation_candidates", {})
            rotation_signals = rotation_candidates.get("rotation_signals", {})
            
            # Process rotation signals
            buy_signals = []
            sell_signals = []
            
            for symbol, signal_data in rotation_signals.items():
                signal = signal_data.get("signal", "hold")
                strength = signal_data.get("strength", 0)
                sector = signal_data.get("sector", "Unknown")
                
                if signal in ["strong_buy", "buy"]:
                    buy_signals.append({
                        "symbol": symbol,
                        "sector": sector,
                        "signal": signal,
                        "strength": strength,
                        "rationale": signal_data.get("rationale", "")
                    })
                elif signal in ["strong_sell", "sell"]:
                    sell_signals.append({
                        "symbol": symbol,
                        "sector": sector,
                        "signal": signal,
                        "strength": strength,
                        "rationale": signal_data.get("rationale", "")
                    })
            
            # Sort by signal strength
            recommendations["top_sectors_to_buy"] = sorted(
                buy_signals, key=lambda x: x["strength"], reverse=True
            )[:5]  # Top 5 buy recommendations
            
            recommendations["sectors_to_reduce"] = sorted(
                sell_signals, key=lambda x: x["strength"], reverse=True
            )[:5]  # Top 5 sell recommendations
            
            # Identify momentum opportunities
            momentum_plays = rotation_candidates.get("momentum_plays", [])
            recommendations["momentum_opportunities"] = momentum_plays[:3]  # Top 3 momentum plays
            
            # Determine overall market sentiment
            total_buy_strength = sum(s["strength"] for s in buy_signals)
            total_sell_strength = sum(s["strength"] for s in sell_signals)
            
            if total_buy_strength > total_sell_strength * 1.5:
                recommendations["overall_market_sentiment"] = "bullish"
            elif total_sell_strength > total_buy_strength * 1.5:
                recommendations["overall_market_sentiment"] = "bearish"
            else:
                recommendations["overall_market_sentiment"] = "neutral"
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating rotation recommendations: {e}")
            return {}
    
    def _optimize_sector_allocation(self, sector_analysis: Dict[str, Any], 
                                  current_allocation: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Optimize sector allocation based on analysis."""
        try:
            optimization = {
                "recommended_allocation": {},
                "allocation_changes": {},
                "rebalancing_required": False,
                "expected_improvement": 0.0,
                "allocation_rationale": {}
            }
            
            # Get sector performance data
            sector_performance = sector_analysis.get("sector_performance", {})
            rankings = sector_analysis.get("sector_rankings", {})
            
            if not sector_performance:
                return optimization
            
            # Simple equal-weight optimization with momentum and risk adjustments
            valid_sectors = []
            for symbol, perf in sector_performance.items():
                if "error" not in perf and isinstance(perf, dict):
                    medium_return = perf.get("period_returns", {}).get("medium_term", 0)
                    momentum = perf.get("momentum_indicators", {}).get("roc_20_day", 0)
                    volatility = perf.get("volatility_metrics", {}).get("volatility_20d", 0.2)
                    
                    # Calculate allocation score
                    score = (
                        medium_return * 0.4 +
                        momentum * 0.4 -
                        volatility * 0.2  # Penalty for high volatility
                    )
                    
                    valid_sectors.append({
                        "symbol": symbol,
                        "sector": perf.get("sector", "Unknown"),
                        "score": score,
                        "return": medium_return,
                        "momentum": momentum,
                        "volatility": volatility
                    })
            
            if not valid_sectors:
                return optimization
            
            # Sort by score and create allocation
            valid_sectors.sort(key=lambda x: x["score"], reverse=True)
            
            # Allocate more to top performers, but maintain diversification
            total_allocation = 1.0
            num_sectors = min(len(valid_sectors), 8)  # Max 8 sectors for diversification
            
            # Weight allocation based on scores
            scores = [s["score"] for s in valid_sectors[:num_sectors]]
            min_score = min(scores)
            adjusted_scores = [s - min_score + 0.1 for s in scores]  # Ensure positive weights
            total_score = sum(adjusted_scores)
            
            for i, sector in enumerate(valid_sectors[:num_sectors]):
                weight = (adjusted_scores[i] / total_score) * total_allocation
                weight = max(0.05, min(0.25, weight))  # Min 5%, max 25% per sector
                
                optimization["recommended_allocation"][sector["symbol"]] = weight
                optimization["allocation_rationale"][sector["symbol"]] = (
                    f"Score: {sector['score']:.3f} "
                    f"(Return: {sector['return']:.1%}, "
                    f"Momentum: {sector['momentum']:.1%}, "
                    f"Vol: {sector['volatility']:.1%})"
                )
            
            # Normalize allocations to sum to 1.0
            total_weight = sum(optimization["recommended_allocation"].values())
            if total_weight > 0:
                for symbol in optimization["recommended_allocation"]:
                    optimization["recommended_allocation"][symbol] /= total_weight
            
            # Calculate allocation changes if current allocation provided
            if current_allocation:
                for symbol in optimization["recommended_allocation"]:
                    current = current_allocation.get(symbol, 0)
                    recommended = optimization["recommended_allocation"][symbol]
                    change = recommended - current
                    
                    if abs(change) > 0.02:  # 2% threshold for meaningful change
                        optimization["allocation_changes"][symbol] = {
                            "current": current,
                            "recommended": recommended,
                            "change": change,
                            "change_type": "increase" if change > 0 else "decrease"
                        }
                
                optimization["rebalancing_required"] = len(optimization["allocation_changes"]) > 0
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing sector allocation: {e}")
            return {}
    
    def _add_risk_management(self, sector_analysis: Dict[str, Any], 
                           portfolio_optimization: Dict[str, Any]) -> Dict[str, Any]:
        """Add risk management considerations to the strategy."""
        try:
            risk_mgmt = {
                "concentration_risk": {},
                "correlation_risk": {},
                "volatility_risk": {},
                "drawdown_protection": {},
                "risk_budget": {},
                "hedging_suggestions": []
            }
            
            recommended_allocation = portfolio_optimization.get("recommended_allocation", {})
            risk_metrics = sector_analysis.get("risk_metrics", {})
            
            # Analyze concentration risk
            if recommended_allocation:
                max_allocation = max(recommended_allocation.values())
                max_symbol = max(recommended_allocation, key=recommended_allocation.get)
                
                risk_mgmt["concentration_risk"] = {
                    "max_single_allocation": max_allocation,
                    "max_allocation_symbol": max_symbol,
                    "concentration_level": "high" if max_allocation > 0.3 else "medium" if max_allocation > 0.2 else "low",
                    "diversification_score": 1.0 - max_allocation
                }
            
            # Analyze correlation risk
            correlations = risk_metrics.get("sector_correlations", {})
            if correlations:
                high_correlations = []
                for symbol1, corr_dict in correlations.items():
                    for symbol2, corr_value in corr_dict.items():
                        if symbol1 < symbol2 and abs(corr_value) > 0.7:  # High correlation threshold
                            high_correlations.append({
                                "pair": f"{symbol1}-{symbol2}",
                                "correlation": corr_value,
                                "risk_level": "high" if abs(corr_value) > 0.8 else "medium"
                            })
                
                risk_mgmt["correlation_risk"] = {
                    "high_correlation_pairs": high_correlations,
                    "correlation_risk_level": "high" if len(high_correlations) > 3 else "medium" if len(high_correlations) > 1 else "low"
                }
            
            # Add hedging suggestions
            risk_mgmt["hedging_suggestions"] = [
                {
                    "strategy": "defensive_sectors",
                    "description": "Consider allocation to defensive sectors (utilities, consumer staples) during market stress",
                    "instruments": ["XLU", "XLP"],
                    "allocation": "5-10%"
                },
                {
                    "strategy": "inverse_etfs",
                    "description": "Use inverse ETFs for hedging during high volatility periods",
                    "instruments": ["SH", "PSQ"],
                    "allocation": "2-5%"
                }
            ]
            
            return risk_mgmt
            
        except Exception as e:
            logger.error(f"Error adding risk management: {e}")
            return {}
    
    def _create_execution_plan(self, rotation_recommendations: Dict[str, Any],
                             portfolio_optimization: Dict[str, Any],
                             current_allocation: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """Create detailed execution plan for the rotation strategy."""
        try:
            execution_plan = {
                "execution_priority": [],
                "trade_sizing": {},
                "execution_timeline": {},
                "cost_considerations": {},
                "monitoring_plan": {}
            }
            
            allocation_changes = portfolio_optimization.get("allocation_changes", {})
            
            # Create execution priority list
            trades = []
            for symbol, change_data in allocation_changes.items():
                change = change_data.get("change", 0)
                change_type = change_data.get("change_type", "hold")
                
                # Prioritize sells before buys for better execution
                priority = 1 if change_type == "decrease" else 2
                
                trades.append({
                    "symbol": symbol,
                    "action": "sell" if change_type == "decrease" else "buy",
                    "allocation_change": abs(change),
                    "priority": priority,
                    "urgency": "high" if abs(change) > 0.1 else "medium" if abs(change) > 0.05 else "low"
                })
            
            execution_plan["execution_priority"] = sorted(trades, key=lambda x: (x["priority"], -x["allocation_change"]))
            
            # Add execution timeline
            execution_plan["execution_timeline"] = {
                "immediate": [t for t in trades if t["urgency"] == "high"],
                "within_week": [t for t in trades if t["urgency"] == "medium"],
                "within_month": [t for t in trades if t["urgency"] == "low"]
            }
            
            # Add monitoring plan
            execution_plan["monitoring_plan"] = {
                "daily_monitoring": ["price movements", "volume patterns", "news events"],
                "weekly_review": ["performance attribution", "allocation drift", "risk metrics"],
                "monthly_rebalancing": ["full strategy review", "allocation optimization", "risk assessment"],
                "trigger_events": ["market volatility spike", "sector rotation signals", "correlation breakdown"]
            }
            
            return execution_plan
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            return {}