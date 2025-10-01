"""Technical analysis engine with multi-timeframe indicator calculations."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculates various technical indicators."""

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of close prices
            period: RSI period (default 14)
            
        Returns:
            RSI values as pandas Series
        """
        if len(prices) < period + 1:
            logger.warning(f"Insufficient data for RSI calculation. Need {period + 1}, got {len(prices)}")
            return pd.Series(dtype=float)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)  # Fill NaN with neutral RSI

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Series of close prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line EMA period (default 9)
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        if len(prices) < max(fast, slow, signal) + 1:
            logger.warning(f"Insufficient data for MACD calculation")
            return {
                "macd": pd.Series(dtype=float),
                "signal": pd.Series(dtype=float),
                "histogram": pd.Series(dtype=float)
            }
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line.fillna(0.0),
            "signal": signal_line.fillna(0.0),
            "histogram": histogram.fillna(0.0)
        }

    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands.
        
        Args:
            prices: Series of close prices
            period: Moving average period (default 20)
            std_dev: Standard deviation multiplier (default 2.0)
            
        Returns:
            Dictionary with upper, middle (SMA), and lower bands
        """
        if len(prices) < period:
            logger.warning(f"Insufficient data for Bollinger Bands calculation")
            return {
                "upper": pd.Series(dtype=float),
                "middle": pd.Series(dtype=float),
                "lower": pd.Series(dtype=float)
            }
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return {
            "upper": upper.bfill(),
            "middle": sma.bfill(),
            "lower": lower.bfill()
        }

    @staticmethod
    def moving_averages(prices: pd.Series, periods: List[int] = [5, 10, 20, 50, 200]) -> Dict[str, pd.Series]:
        """Calculate Simple Moving Averages for multiple periods.
        
        Args:
            prices: Series of close prices
            periods: List of MA periods
            
        Returns:
            Dictionary with SMA values for each period
        """
        mas = {}
        for period in periods:
            if len(prices) >= period:
                mas[f"sma_{period}"] = prices.rolling(window=period).mean().bfill()
            else:
                logger.warning(f"Insufficient data for SMA_{period}")
                mas[f"sma_{period}"] = pd.Series(dtype=float)
        
        return mas

    @staticmethod
    def exponential_moving_averages(prices: pd.Series, periods: List[int] = [12, 26, 50]) -> Dict[str, pd.Series]:
        """Calculate Exponential Moving Averages for multiple periods.
        
        Args:
            prices: Series of close prices
            periods: List of EMA periods
            
        Returns:
            Dictionary with EMA values for each period
        """
        emas = {}
        for period in periods:
            if len(prices) >= period:
                emas[f"ema_{period}"] = prices.ewm(span=period).mean().bfill()
            else:
                logger.warning(f"Insufficient data for EMA_{period}")
                emas[f"ema_{period}"] = pd.Series(dtype=float)
        
        return emas

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            k_period: %K period (default 14)
            d_period: %D smoothing period (default 3)
            
        Returns:
            Dictionary with %K and %D values
        """
        if len(close) < k_period:
            logger.warning(f"Insufficient data for Stochastic calculation")
            return {
                "k_percent": pd.Series(dtype=float),
                "d_percent": pd.Series(dtype=float)
            }
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            "k_percent": k_percent.fillna(50.0),
            "d_percent": d_percent.fillna(50.0)
        }

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR).
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period (default 14)
            
        Returns:
            ATR values as pandas Series
        """
        if len(close) < period + 1:
            logger.warning(f"Insufficient data for ATR calculation")
            return pd.Series(dtype=float)
        
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        # Create DataFrame from Series for proper concat
        tr_df = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
        true_range = tr_df.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.bfill()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index (ADX) and directional indicators.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ADX period (default 14)
            
        Returns:
            Dictionary with ADX, +DI, and -DI values
        """
        if len(close) < period * 2:
            logger.warning(f"Insufficient data for ADX calculation")
            return {
                "adx": pd.Series(dtype=float),
                "plus_di": pd.Series(dtype=float),
                "minus_di": pd.Series(dtype=float)
            }
        
        # Calculate directional movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=close.index)
        minus_dm = pd.Series(minus_dm, index=close.index)
        
        # Calculate True Range
        atr_values = TechnicalIndicators.atr(high, low, close, period)
        
        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_values)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_values)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return {
            "adx": adx.fillna(25.0),
            "plus_di": plus_di.fillna(25.0),
            "minus_di": minus_di.fillna(25.0)
        }


class TimeframeManager:
    """Manages multiple timeframes and data synchronization."""
    
    SUPPORTED_TIMEFRAMES = {
        "1m": {"minutes": 1, "label": "1 Minute"},
        "5m": {"minutes": 5, "label": "5 Minutes"},
        "15m": {"minutes": 15, "label": "15 Minutes"},
        "1h": {"minutes": 60, "label": "1 Hour"},
        "4h": {"minutes": 240, "label": "4 Hours"},
        "1d": {"minutes": 1440, "label": "1 Day"}
    }
    
    def __init__(self):
        """Initialize timeframe manager."""
        self.timeframes = list(self.SUPPORTED_TIMEFRAMES.keys())
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is supported.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            True if supported, False otherwise
        """
        return timeframe in self.SUPPORTED_TIMEFRAMES
    
    def get_timeframe_minutes(self, timeframe: str) -> int:
        """Get minutes for a timeframe.
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            Number of minutes in the timeframe
        """
        if not self.validate_timeframe(timeframe):
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        return self.SUPPORTED_TIMEFRAMES[timeframe]["minutes"]
    
    def resample_data(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample data to target timeframe.
        
        Args:
            data: OHLCV data with datetime index
            target_timeframe: Target timeframe for resampling
            
        Returns:
            Resampled DataFrame
        """
        if not self.validate_timeframe(target_timeframe):
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
        
        if data.empty:
            return data
        
        # Map timeframe to pandas offset
        timeframe_map = {
            "1m": "1min",
            "5m": "5min", 
            "15m": "15min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1D"
        }
        
        freq = timeframe_map[target_timeframe]
        
        # Ensure data has proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)
            else:
                logger.error("Data must have datetime index or timestamp column")
                return pd.DataFrame()
        
        # Resample OHLCV data
        resampled = data.resample(freq).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled
    
    def synchronize_timeframes(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Synchronize data across multiple timeframes to same time range.
        
        Args:
            data_dict: Dictionary with timeframe as key and DataFrame as value
            
        Returns:
            Synchronized data dictionary
        """
        if not data_dict:
            return {}
        
        # Find common time range across all timeframes
        start_times = []
        end_times = []
        
        for timeframe, data in data_dict.items():
            if not data.empty:
                start_times.append(data.index.min())
                end_times.append(data.index.max())
        
        if not start_times:
            return data_dict
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        # Filter each timeframe to common range
        synchronized_data = {}
        for timeframe, data in data_dict.items():
            if not data.empty:
                synchronized_data[timeframe] = data.loc[common_start:common_end]
            else:
                synchronized_data[timeframe] = data
        
        return synchronized_data


class MultiTimeframeAnalysis:
    """Performs technical analysis across multiple timeframes."""
    
    def __init__(self):
        """Initialize multi-timeframe analysis engine."""
        self.indicators = TechnicalIndicators()
        self.timeframe_manager = TimeframeManager()
        
    def calculate_indicators(self, data: pd.DataFrame, timeframes: List[str] = None) -> Dict[str, Dict]:
        """Calculate technical indicators across multiple timeframes.
        
        Args:
            data: OHLCV data with datetime index
            timeframes: List of timeframes to analyze (default: ['5m', '1h', '1d'])
            
        Returns:
            Dictionary with timeframe as key and indicators as value
        """
        if timeframes is None:
            timeframes = ['5m', '1h', '1d']
        
        logger.info(f"Calculating indicators for timeframes: {timeframes}")
        
        if data.empty:
            logger.warning("Empty data provided for indicator calculation")
            return {}
        
        results = {}
        
        for timeframe in timeframes:
            try:
                # Resample data to target timeframe
                if timeframe == "1m":
                    # Use original data if it's already 1-minute
                    tf_data = data
                else:
                    tf_data = self.timeframe_manager.resample_data(data, timeframe)
                
                if tf_data.empty:
                    logger.warning(f"No data available for timeframe {timeframe}")
                    continue
                
                # Calculate indicators for this timeframe
                indicators = self._calculate_timeframe_indicators(tf_data)
                indicators["timeframe"] = timeframe
                indicators["data_points"] = len(tf_data)
                indicators["latest_timestamp"] = tf_data.index[-1].isoformat()
                
                results[timeframe] = indicators
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {timeframe}: {e}")
                results[timeframe] = {"error": str(e)}
        
        return results
    
    def _calculate_timeframe_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all indicators for a specific timeframe.
        
        Args:
            data: OHLCV data for the timeframe
            
        Returns:
            Dictionary with all calculated indicators
        """
        if len(data) < 50:  # Minimum data points for meaningful analysis
            logger.warning(f"Limited data available: {len(data)} points")
        
        indicators = {}
        
        try:
            # Price series
            close = data['Close']
            high = data['High']
            low = data['Low']
            open_price = data['Open']
            volume = data['Volume']
            
            # Basic price info
            indicators["latest_price"] = float(close.iloc[-1])
            indicators["price_change"] = float(close.iloc[-1] - close.iloc[-2]) if len(close) > 1 else 0.0
            indicators["price_change_pct"] = (indicators["price_change"] / close.iloc[-2] * 100) if len(close) > 1 and close.iloc[-2] != 0 else 0.0
            
            # RSI
            rsi = self.indicators.rsi(close)
            if not rsi.empty:
                indicators["rsi"] = {
                    "current": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
                    "previous": float(rsi.iloc[-2]) if len(rsi) > 1 and not pd.isna(rsi.iloc[-2]) else 50.0,
                    "overbought": rsi.iloc[-1] > 70 if not pd.isna(rsi.iloc[-1]) else False,
                    "oversold": rsi.iloc[-1] < 30 if not pd.isna(rsi.iloc[-1]) else False
                }
            
            # MACD
            macd_result = self.indicators.macd(close)
            if not macd_result["macd"].empty:
                indicators["macd"] = {
                    "macd": float(macd_result["macd"].iloc[-1]) if not pd.isna(macd_result["macd"].iloc[-1]) else 0.0,
                    "signal": float(macd_result["signal"].iloc[-1]) if not pd.isna(macd_result["signal"].iloc[-1]) else 0.0,
                    "histogram": float(macd_result["histogram"].iloc[-1]) if not pd.isna(macd_result["histogram"].iloc[-1]) else 0.0,
                    "bullish": macd_result["macd"].iloc[-1] > macd_result["signal"].iloc[-1] if not pd.isna(macd_result["macd"].iloc[-1]) else False
                }
            
            # Bollinger Bands
            bb_result = self.indicators.bollinger_bands(close)
            if not bb_result["middle"].empty:
                indicators["bollinger_bands"] = {
                    "upper": float(bb_result["upper"].iloc[-1]) if not pd.isna(bb_result["upper"].iloc[-1]) else indicators["latest_price"] * 1.02,
                    "middle": float(bb_result["middle"].iloc[-1]) if not pd.isna(bb_result["middle"].iloc[-1]) else indicators["latest_price"],
                    "lower": float(bb_result["lower"].iloc[-1]) if not pd.isna(bb_result["lower"].iloc[-1]) else indicators["latest_price"] * 0.98,
                    "position": self._get_bb_position(indicators["latest_price"], bb_result)
                }
            
            # Moving Averages
            sma_result = self.indicators.moving_averages(close)
            ema_result = self.indicators.exponential_moving_averages(close)
            
            indicators["moving_averages"] = {}
            for key, sma in sma_result.items():
                if not sma.empty and not pd.isna(sma.iloc[-1]):
                    indicators["moving_averages"][key] = float(sma.iloc[-1])
            
            for key, ema in ema_result.items():
                if not ema.empty and not pd.isna(ema.iloc[-1]):
                    indicators["moving_averages"][key] = float(ema.iloc[-1])
            
            # Stochastic
            stoch_result = self.indicators.stochastic(high, low, close)
            if not stoch_result["k_percent"].empty:
                indicators["stochastic"] = {
                    "k_percent": float(stoch_result["k_percent"].iloc[-1]) if not pd.isna(stoch_result["k_percent"].iloc[-1]) else 50.0,
                    "d_percent": float(stoch_result["d_percent"].iloc[-1]) if not pd.isna(stoch_result["d_percent"].iloc[-1]) else 50.0,
                    "overbought": stoch_result["k_percent"].iloc[-1] > 80 if not pd.isna(stoch_result["k_percent"].iloc[-1]) else False,
                    "oversold": stoch_result["k_percent"].iloc[-1] < 20 if not pd.isna(stoch_result["k_percent"].iloc[-1]) else False
                }
            
            # ATR
            atr_result = self.indicators.atr(high, low, close)
            if not atr_result.empty:
                indicators["atr"] = {
                    "current": float(atr_result.iloc[-1]) if not pd.isna(atr_result.iloc[-1]) else 0.0,
                    "percentage": (float(atr_result.iloc[-1]) / indicators["latest_price"] * 100) if not pd.isna(atr_result.iloc[-1]) and indicators["latest_price"] > 0 else 0.0
                }
            
            # ADX
            adx_result = self.indicators.adx(high, low, close)
            if not adx_result["adx"].empty:
                indicators["adx"] = {
                    "adx": float(adx_result["adx"].iloc[-1]) if not pd.isna(adx_result["adx"].iloc[-1]) else 25.0,
                    "plus_di": float(adx_result["plus_di"].iloc[-1]) if not pd.isna(adx_result["plus_di"].iloc[-1]) else 25.0,
                    "minus_di": float(adx_result["minus_di"].iloc[-1]) if not pd.isna(adx_result["minus_di"].iloc[-1]) else 25.0,
                    "trend_strength": self._get_adx_strength(float(adx_result["adx"].iloc[-1]) if not pd.isna(adx_result["adx"].iloc[-1]) else 25.0)
                }
            
            # Volume analysis
            if len(volume) > 20:
                avg_volume = volume.tail(20).mean()
                indicators["volume"] = {
                    "current": int(volume.iloc[-1]),
                    "average_20": int(avg_volume),
                    "relative": float(volume.iloc[-1] / avg_volume) if avg_volume > 0 else 1.0,
                    "high_volume": volume.iloc[-1] > avg_volume * 1.5 if avg_volume > 0 else False
                }
            
        except Exception as e:
            logger.error(f"Error in indicator calculation: {e}")
            indicators["calculation_error"] = str(e)
        
        return indicators
    
    def _get_bb_position(self, price: float, bb_result: Dict[str, pd.Series]) -> str:
        """Determine price position relative to Bollinger Bands."""
        try:
            upper = bb_result["upper"].iloc[-1]
            lower = bb_result["lower"].iloc[-1]
            
            if pd.isna(upper) or pd.isna(lower):
                return "unknown"
            
            if price >= upper:
                return "above_upper"
            elif price <= lower:
                return "below_lower"
            else:
                return "middle"
        except:
            return "unknown"
    
    def _get_adx_strength(self, adx_value: float) -> str:
        """Determine trend strength from ADX value."""
        if adx_value >= 50:
            return "very_strong"
        elif adx_value >= 25:
            return "strong"
        elif adx_value >= 20:
            return "moderate"
        else:
            return "weak"
    
    def generate_signals(self, indicators: Dict[str, Dict]) -> Dict:
        """Generate trading signals from multi-timeframe indicators.
        
        Args:
            indicators: Multi-timeframe indicator results
            
        Returns:
            Dictionary with signals and analysis
        """
        logger.info("Generating trading signals from multi-timeframe analysis")
        
        signals = {
            "timestamp": datetime.now().isoformat(),
            "overall_signal": "HOLD",
            "confidence": 0.5,
            "timeframe_signals": {},
            "confluence_factors": [],
            "risk_factors": []
        }
        
        timeframe_scores = {}
        
        # Analyze each timeframe
        for timeframe, tf_indicators in indicators.items():
            if "error" in tf_indicators:
                continue
                
            score = self._calculate_timeframe_signal_score(tf_indicators)
            timeframe_scores[timeframe] = score
            
            signals["timeframe_signals"][timeframe] = {
                "signal": "BUY" if score > 0.6 else "SELL" if score < 0.4 else "HOLD",
                "score": score,
                "key_factors": self._get_key_factors(tf_indicators)
            }
        
        # Calculate overall signal with timeframe weighting
        if timeframe_scores:
            # Weight longer timeframes more heavily
            weights = {"1m": 0.1, "5m": 0.2, "15m": 0.3, "1h": 0.4, "4h": 0.6, "1d": 1.0}
            
            weighted_score = 0
            total_weight = 0
            
            for timeframe, score in timeframe_scores.items():
                weight = weights.get(timeframe, 0.5)
                weighted_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                overall_score = weighted_score / total_weight
                signals["confidence"] = abs(overall_score - 0.5) * 2  # 0 to 1 scale
                
                if overall_score > 0.6:
                    signals["overall_signal"] = "BUY"
                elif overall_score < 0.4:
                    signals["overall_signal"] = "SELL"
                else:
                    signals["overall_signal"] = "HOLD"
        
        # Identify confluence factors
        signals["confluence_factors"] = self._identify_confluence(indicators)
        signals["risk_factors"] = self._identify_risks(indicators)
        
        return signals
    
    def _calculate_timeframe_signal_score(self, indicators: Dict) -> float:
        """Calculate signal score for a specific timeframe."""
        score = 0.5  # Neutral starting point
        factor_count = 0
        
        try:
            # RSI factor
            if "rsi" in indicators:
                rsi_current = indicators["rsi"]["current"]
                if rsi_current < 30:
                    score += 0.2  # Oversold, bullish
                elif rsi_current > 70:
                    score -= 0.2  # Overbought, bearish
                factor_count += 1
            
            # MACD factor
            if "macd" in indicators:
                if indicators["macd"]["bullish"]:
                    score += 0.15
                else:
                    score -= 0.15
                factor_count += 1
            
            # Moving average factor
            if "moving_averages" in indicators:
                mas = indicators["moving_averages"]
                price = indicators["latest_price"]
                
                above_count = 0
                below_count = 0
                
                for ma_name, ma_value in mas.items():
                    if price > ma_value:
                        above_count += 1
                    else:
                        below_count += 1
                
                total_mas = above_count + below_count
                if total_mas > 0:
                    ma_score = above_count / total_mas
                    score += (ma_score - 0.5) * 0.3
                    factor_count += 1
            
            # Stochastic factor
            if "stochastic" in indicators:
                if indicators["stochastic"]["oversold"]:
                    score += 0.1
                elif indicators["stochastic"]["overbought"]:
                    score -= 0.1
                factor_count += 1
            
            # Volume factor
            if "volume" in indicators:
                if indicators["volume"]["high_volume"]:
                    # High volume confirms the direction
                    if score > 0.5:
                        score += 0.05
                    elif score < 0.5:
                        score -= 0.05
                factor_count += 1
            
        except Exception as e:
            logger.warning(f"Error calculating signal score: {e}")
        
        # Ensure score stays in valid range
        return max(0.0, min(1.0, score))
    
    def _get_key_factors(self, indicators: Dict) -> List[str]:
        """Identify key factors for the timeframe signal."""
        factors = []
        
        try:
            if "rsi" in indicators:
                rsi = indicators["rsi"]["current"]
                if rsi > 70:
                    factors.append("RSI overbought")
                elif rsi < 30:
                    factors.append("RSI oversold")
            
            if "macd" in indicators:
                if indicators["macd"]["bullish"]:
                    factors.append("MACD bullish crossover")
                else:
                    factors.append("MACD bearish crossover")
            
            if "bollinger_bands" in indicators:
                position = indicators["bollinger_bands"]["position"]
                if position == "above_upper":
                    factors.append("Price above upper Bollinger Band")
                elif position == "below_lower":
                    factors.append("Price below lower Bollinger Band")
            
            if "volume" in indicators and indicators["volume"]["high_volume"]:
                factors.append("High volume confirmation")
            
        except Exception as e:
            logger.warning(f"Error identifying key factors: {e}")
        
        return factors
    
    def _identify_confluence(self, indicators: Dict[str, Dict]) -> List[str]:
        """Identify confluence factors across timeframes."""
        confluence = []
        
        try:
            # Check if multiple timeframes agree
            bullish_count = 0
            bearish_count = 0
            
            for timeframe, tf_indicators in indicators.items():
                if "error" in tf_indicators:
                    continue
                
                # Simple confluence check based on RSI and MACD
                if "rsi" in tf_indicators and "macd" in tf_indicators:
                    rsi = tf_indicators["rsi"]["current"]
                    macd_bullish = tf_indicators["macd"]["bullish"]
                    
                    if rsi < 40 and macd_bullish:
                        bullish_count += 1
                    elif rsi > 60 and not macd_bullish:
                        bearish_count += 1
            
            if bullish_count >= 2:
                confluence.append("Multi-timeframe bullish confluence")
            if bearish_count >= 2:
                confluence.append("Multi-timeframe bearish confluence")
            
        except Exception as e:
            logger.warning(f"Error identifying confluence: {e}")
        
        return confluence
    
    def _identify_risks(self, indicators: Dict[str, Dict]) -> List[str]:
        """Identify risk factors from indicators."""
        risks = []
        
        try:
            for timeframe, tf_indicators in indicators.items():
                if "error" in tf_indicators:
                    continue
                
                # High volatility risk
                if "atr" in tf_indicators:
                    atr_pct = tf_indicators["atr"]["percentage"]
                    if atr_pct > 5.0:  # More than 5% daily range
                        risks.append(f"High volatility in {timeframe} timeframe")
                
                # Overbought/oversold extremes
                if "rsi" in tf_indicators:
                    rsi = tf_indicators["rsi"]["current"]
                    if rsi > 80:
                        risks.append(f"Extreme overbought conditions in {timeframe}")
                    elif rsi < 20:
                        risks.append(f"Extreme oversold conditions in {timeframe}")
                
                # Weak trend strength
                if "adx" in tf_indicators:
                    adx = tf_indicators["adx"]["trend_strength"]
                    if adx == "weak":
                        risks.append(f"Weak trend strength in {timeframe}")
        
        except Exception as e:
            logger.warning(f"Error identifying risks: {e}")
        
        return risks