"""Resonance Engine for signal consensus and market alignment analysis."""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class ConsensusLevel(Enum):
    """Consensus confidence levels."""
    VERY_HIGH = "very_high"
    HIGH = "high" 
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"

class AlignmentStatus(Enum):
    """Signal alignment status."""
    FULLY_ALIGNED = "fully_aligned"
    MOSTLY_ALIGNED = "mostly_aligned"
    PARTIALLY_ALIGNED = "partially_aligned"
    CONFLICTED = "conflicted"
    NO_CONSENSUS = "no_consensus"

class ResonanceEngine:
    """Multi-dimensional signal consensus engine using resonance theory."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ResonanceEngine with optional configuration."""
        self.config = config or {}
        self.consensus_threshold = self.config.get('consensus_threshold', 0.7)
        self.alignment_threshold = self.config.get('alignment_threshold', 0.6)
        self.signal_weights = self.config.get('signal_weights', {
            'technical': 0.4, 'fundamental': 0.3, 'sentiment': 0.2, 'volume': 0.1
        })
    
    def calculate_consensus(self, multi_timeframe_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate signal consensus across multiple timeframes and analysis types.
        
        Args:
            multi_timeframe_data: Multi-timeframe analysis results
            
        Returns:
            Dict with consensus metrics
        """
        try:
            # Import shared utilities
            from ..utils.shared import normalize_signals, calculate_agreement_ratio, validate_signal_quality
            
            if not multi_timeframe_data or 'timeframes' not in multi_timeframe_data:
                return self._get_fallback_consensus()
            
            timeframes = multi_timeframe_data['timeframes']
            
            # Extract signals from all timeframes
            all_signals = []
            signal_strengths = {}
            timeframe_weights = self._calculate_timeframe_weights(timeframes)
            
            for tf_name, tf_data in timeframes.items():
                tf_weight = timeframe_weights.get(tf_name, 1.0)
                
                # Extract technical signals
                technical_signals = self._extract_technical_signals(tf_data)
                if technical_signals:
                    all_signals.extend(technical_signals)
                    signal_strengths.update({f"tech_{tf_name}": len(technical_signals) * tf_weight})
                
                # Extract fundamental signals
                fundamental_signals = self._extract_fundamental_signals(tf_data)
                if fundamental_signals:
                    all_signals.extend(fundamental_signals)
                    signal_strengths.update({f"fund_{tf_name}": len(fundamental_signals) * tf_weight})
                
                # Extract sentiment signals
                sentiment_signals = self._extract_sentiment_signals(tf_data)
                if sentiment_signals:
                    all_signals.extend(sentiment_signals)
                    signal_strengths.update({f"sent_{tf_name}": len(sentiment_signals) * tf_weight})
            
            # Calculate agreement ratio
            agreement_ratio = calculate_agreement_ratio(all_signals) if all_signals else 0.5
            
            # Normalize signal strengths
            normalized_strengths = normalize_signals(signal_strengths) if signal_strengths else {}
            
            # Calculate consensus score
            consensus_score = self._calculate_consensus_score(agreement_ratio, normalized_strengths, timeframe_weights)
            
            # Determine confidence level
            confidence_level = self._determine_confidence_level(consensus_score, len(all_signals))
            
            # Assess alignment status
            alignment_status = self._assess_alignment_status(all_signals, agreement_ratio)
            
            # Validate overall signal quality
            signal_age_minutes = 5  # Assume recent signals
            overall_signal = {
                'strength': consensus_score,
                'confidence': agreement_ratio
            }
            signal_validation = validate_signal_quality(overall_signal, signal_age_minutes)
            
            return {
                'consensus_score': round(consensus_score, 3),
                'confidence_level': confidence_level.value,
                'alignment_status': alignment_status.value,
                'agreement_ratio': round(agreement_ratio, 3),
                'signal_count': len(all_signals),
                'timeframe_weights': timeframe_weights,
                'signal_strengths': normalized_strengths,
                'validation': signal_validation,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"Consensus calculation failed: {e}")
            return self._get_fallback_consensus()
    
    def _calculate_timeframe_weights(self, timeframes: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weights for different timeframes based on reliability."""
        try:
            priorities = {'1d': 1.0, '4h': 0.8, '1h': 0.6, '15m': 0.4, '5m': 0.3, '1m': 0.2}
            weights = {}
            for tf_name in timeframes.keys():
                tf_key = tf_name.replace('_', '').replace('min', 'm').replace('hour', 'h').replace('day', 'd')
                weights[tf_name] = priorities.get(tf_key, 0.5)
            return weights
        except Exception:
            return {tf: 0.5 for tf in timeframes.keys()}
    
    def _extract_technical_signals(self, timeframe_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract technical analysis signals from timeframe data."""
        signals = []
        try:
            technical = timeframe_data.get('technical', {})
            # RSI signals
            if 'rsi' in technical and isinstance(technical['rsi'], (int, float)):
                rsi_val = technical['rsi']
                if rsi_val > 70: signals.append({'direction': 'bearish', 'type': 'rsi_overbought', 'strength': (rsi_val - 70) / 30})
                elif rsi_val < 30: signals.append({'direction': 'bullish', 'type': 'rsi_oversold', 'strength': (30 - rsi_val) / 30})
            # MACD signals
            macd = technical.get('macd', {})
            if isinstance(macd, dict) and 'crossover' in macd:
                if macd['crossover'] == 'bullish': signals.append({'direction': 'bullish', 'type': 'macd_cross', 'strength': 0.7})
                elif macd['crossover'] == 'bearish': signals.append({'direction': 'bearish', 'type': 'macd_cross', 'strength': 0.7})
            # Bollinger Bands signals
            bb = technical.get('bollinger', {})
            if isinstance(bb, dict) and 'position' in bb:
                if bb['position'] == 'above_upper': signals.append({'direction': 'bearish', 'type': 'bb_overbought', 'strength': 0.6})
                elif bb['position'] == 'below_lower': signals.append({'direction': 'bullish', 'type': 'bb_oversold', 'strength': 0.6})
            # Trend signals
            trend = technical.get('trend', {})
            if isinstance(trend, dict) and 'direction' in trend:
                direction = trend['direction']
                if direction in ['up', 'uptrend', 'bullish']: signals.append({'direction': 'bullish', 'type': 'trend', 'strength': 0.8})
                elif direction in ['down', 'downtrend', 'bearish']: signals.append({'direction': 'bearish', 'type': 'trend', 'strength': 0.8})
        except Exception as e:
            logger.debug(f"Technical signal extraction failed: {e}")
        return signals
    
    def _extract_fundamental_signals(self, timeframe_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract fundamental analysis signals from timeframe data."""
        signals = []
        try:
            fundamentals = timeframe_data.get('fundamental', {})
            # P/E ratio signals
            if 'pe_ratio' in fundamentals and isinstance(fundamentals['pe_ratio'], (int, float)):
                pe = fundamentals['pe_ratio']
                if pe < 15: signals.append({'direction': 'bullish', 'type': 'pe_undervalued', 'strength': 0.6})
                elif pe > 25: signals.append({'direction': 'bearish', 'type': 'pe_overvalued', 'strength': 0.6})
            # Growth signals
            if 'growth_score' in fundamentals and isinstance(fundamentals['growth_score'], (int, float)):
                growth = fundamentals['growth_score']
                if growth > 0.7: signals.append({'direction': 'bullish', 'type': 'high_growth', 'strength': growth})
                elif growth < 0.3: signals.append({'direction': 'bearish', 'type': 'low_growth', 'strength': 1 - growth})
        except Exception as e:
            logger.debug(f"Fundamental signal extraction failed: {e}")
        return signals
    
    def _extract_sentiment_signals(self, timeframe_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sentiment analysis signals from timeframe data."""
        signals = []
        try:
            sentiment = timeframe_data.get('sentiment', {})
            # Overall sentiment score
            if 'score' in sentiment and isinstance(sentiment['score'], (int, float)):
                score = sentiment['score']
                if score > 0.2: signals.append({'direction': 'bullish', 'type': 'positive_sentiment', 'strength': score})
                elif score < -0.2: signals.append({'direction': 'bearish', 'type': 'negative_sentiment', 'strength': abs(score)})
            # News sentiment
            if 'news_sentiment' in sentiment and isinstance(sentiment['news_sentiment'], (int, float)):
                news = sentiment['news_sentiment']
                if news > 0.3: signals.append({'direction': 'bullish', 'type': 'positive_news', 'strength': news})
                elif news < -0.3: signals.append({'direction': 'bearish', 'type': 'negative_news', 'strength': abs(news)})
        except Exception as e:
            logger.debug(f"Sentiment signal extraction failed: {e}")
        return signals
    
    def _calculate_consensus_score(self, agreement_ratio: float, signal_strengths: Dict[str, float], 
                                 timeframe_weights: Dict[str, float]) -> float:
        """Calculate overall consensus score combining multiple factors."""
        try:
            base_score = agreement_ratio
            if signal_strengths:
                base_score = (base_score * 0.7) + (np.mean(list(signal_strengths.values())) * 0.3)
            if timeframe_weights:
                base_score = (base_score * 0.8) + (np.mean(list(timeframe_weights.values())) * 0.2)
            return max(0.0, min(1.0, base_score))
        except Exception:
            return 0.5
    
    def _determine_confidence_level(self, consensus_score: float, signal_count: int) -> ConsensusLevel:
        """Determine confidence level based on consensus score and signal count."""
        try:
            adjusted_score = consensus_score * (0.7 + min(1.0, signal_count / 10.0) * 0.3)
            if adjusted_score >= 0.85: return ConsensusLevel.VERY_HIGH
            elif adjusted_score >= 0.7: return ConsensusLevel.HIGH
            elif adjusted_score >= 0.5: return ConsensusLevel.MODERATE
            elif adjusted_score >= 0.3: return ConsensusLevel.LOW
            else: return ConsensusLevel.VERY_LOW
        except Exception:
            return ConsensusLevel.MODERATE
    
    def _assess_alignment_status(self, signals: List[Dict[str, Any]], agreement_ratio: float) -> AlignmentStatus:
        """Assess signal alignment status based on signal directions and agreement."""
        try:
            if not signals: return AlignmentStatus.NO_CONSENSUS
            bullish_count = sum(1 for s in signals if s.get('direction') == 'bullish')
            bearish_count = sum(1 for s in signals if s.get('direction') == 'bearish')
            total_directional = bullish_count + bearish_count
            if total_directional == 0: return AlignmentStatus.NO_CONSENSUS
            directional_ratio = max(bullish_count, bearish_count) / total_directional
            overall_alignment = (directional_ratio * 0.6) + (agreement_ratio * 0.4)
            if overall_alignment >= 0.85: return AlignmentStatus.FULLY_ALIGNED
            elif overall_alignment >= 0.7: return AlignmentStatus.MOSTLY_ALIGNED
            elif overall_alignment >= 0.5: return AlignmentStatus.PARTIALLY_ALIGNED
            elif abs(bullish_count - bearish_count) <= 1: return AlignmentStatus.CONFLICTED
            else: return AlignmentStatus.NO_CONSENSUS
        except Exception:
            return AlignmentStatus.NO_CONSENSUS
    
    def _get_fallback_consensus(self) -> Dict[str, Any]:
        """Get fallback consensus result when calculation fails."""
        return {'consensus_score': 0.5, 'confidence_level': ConsensusLevel.MODERATE.value,
                'alignment_status': AlignmentStatus.NO_CONSENSUS.value, 'agreement_ratio': 0.5,
                'signal_count': 0, 'timeframe_weights': {}, 'signal_strengths': {},
                'validation': {'is_valid': True, 'quality_score': 0.5, 'issues': []},
                'timestamp': datetime.now().isoformat()}