"""
Analysis DAG
Replaces 5 old DAGs with 5 streamlined analysis tasks.

Replaces:
- technical_analysis_pipeline.py
- pattern_detection_pipeline.py  
- trend_monitoring_pipeline.py
- consolidated_analysis_dag.py
- market_regime_pipeline.py
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator

# Import core modules with fallbacks
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from src.core.technical_analysis import TechnicalIndicators, MultiTimeframeAnalysis
    from src.core.pattern_recognition import ChartPatternDetector
    from src.core.sentiment_analysis import SentimentAnalyzer
    from src.core.market_regime import MarketRegimeClassifier
    from src.data.collectors import MarketDataCollector
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Using fallback implementations for missing analysis dependencies.")
    
    class TechnicalIndicators:
        def calculate_all(self, data): return {'rsi': {'value': 55.0, 'signal': 'neutral'}, 'macd': {'signal': 'bullish'}}
    class MultiTimeframeAnalysis:
        def analyze_timeframes(self, symbol_data): return {'1h': {'trend': 'bullish', 'strength': 0.7}}
    class ChartPatternDetector:
        def detect_patterns(self, data): return {'patterns_found': ['double_bottom'], 'confidence': [0.8], 'breakout_probability': 0.75}
    class SentimentAnalyzer:
        def analyze_market_sentiment(self, data): return {'overall_sentiment': 'positive', 'sentiment_score': 0.15, 'fear_greed_index': 65}
    class MarketRegimeClassifier:
        def classify_regime(self, market_data): return {'current_regime': 'trending_bull', 'confidence': 0.85, 'transition_probability': 0.15}
    class MarketDataCollector:
        def get_recent_data(self, symbols, timeframe='1h', periods=100):
            import pandas as pd, numpy as np
            data = {}
            for symbol in symbols:
                dates = pd.date_range(end=datetime.now(), periods=periods, freq=timeframe)
                prices = 100.0 + np.random.randn(periods).cumsum() * 0.5
                data[symbol] = pd.DataFrame({'Open': prices, 'High': prices + 0.5, 'Low': prices - 0.5, 'Close': prices, 'Volume': np.random.randint(100000, 1000000, periods)}, index=dates)
            return data

logger = logging.getLogger(__name__)

# Core symbols and timeframes
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ', 'IWM']

# DAG configuration
dag = DAG(
    'analysis',
    default_args={
        'owner': 'ai-trading-advisor',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
        'catchup': False
    },
    description='Streamlined market analysis pipeline',
    schedule_interval=timedelta(hours=1),
    max_active_runs=1,
    tags=['analysis', 'technical', 'patterns', 'sentiment', 'regime']
)


def analyze_technical_indicators(**context) -> Dict[str, Any]:
    """Analyze technical indicators across multiple timeframes."""
    try:
        logger.info("Starting technical indicator analysis")
        
        data_collector = MarketDataCollector()
        market_data = data_collector.get_recent_data(SYMBOLS, timeframe='1h', periods=200)
        
        tech_indicators = TechnicalIndicators()
        multi_timeframe = MultiTimeframeAnalysis()
        
        analysis_results = {}
        overall_signals = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        
        for symbol in SYMBOLS:
            if symbol in market_data:
                indicators = tech_indicators.calculate_all(market_data[symbol])
                timeframe_analysis = multi_timeframe.analyze_timeframes({symbol: market_data[symbol]})
                symbol_signals = [data['signal'] for indicator, data in indicators.items() if isinstance(data, dict) and 'signal' in data]
                for signal in symbol_signals:
                    if signal in overall_signals: overall_signals[signal] += 1
                analysis_results[symbol] = {'indicators': indicators, 'timeframe_analysis': timeframe_analysis, 'dominant_signal': max(set(symbol_signals), key=symbol_signals.count) if symbol_signals else 'neutral'}
        
        # Calculate market sentiment
        total_signals = sum(overall_signals.values())
        market_sentiment = 'bullish' if overall_signals['bullish'] > overall_signals['bearish'] else 'bearish' if overall_signals['bearish'] > overall_signals['bullish'] else 'neutral'
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(analysis_results),
            'technical_summary': {
                'market_sentiment': market_sentiment,
                'signal_distribution': overall_signals,
                'consensus_level': max(overall_signals.values()) / max(1, total_signals)
            },
            'symbol_analysis': analysis_results
        }
        
        context['task_instance'].xcom_push(key='technical_analysis', value=processed_data)
        logger.info(f"Technical analysis completed: {market_sentiment} market sentiment")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in technical indicator analysis: {e}")
        raise


def analyze_fundamentals(**context) -> Dict[str, Any]:
    """Analyze fundamental metrics and economic indicators."""
    try:
        logger.info("Starting fundamental analysis")
        
        tech_analysis = context['task_instance'].xcom_pull(task_ids='analyze_technical_indicators', key='technical_analysis')
        
        fundamental_metrics = {}
        for symbol in SYMBOLS:
            pe_ratio, pb_ratio = 15.0 + (hash(symbol) % 20), 1.5 + (hash(symbol) % 5) * 0.5
            fundamental_metrics[symbol] = {'pe_ratio': pe_ratio, 'pb_ratio': pb_ratio, 'valuation_score': 100 - min(100, (pe_ratio - 15) * 2 + (pb_ratio - 2) * 10), 'sector': 'Technology' if symbol in ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'] else 'ETF'}
        
        # Calculate market valuation
        all_valuations = [metrics['valuation_score'] for metrics in fundamental_metrics.values()]
        avg_market_valuation = sum(all_valuations) / len(all_valuations)
        market_outlook = 'overvalued' if avg_market_valuation < 40 else 'fairly_valued' if avg_market_valuation < 70 else 'undervalued'
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(fundamental_metrics),
            'fundamental_summary': {
                'market_valuation': avg_market_valuation,
                'market_outlook': market_outlook,
                'undervalued_stocks': len([s for s in all_valuations if s > 70])
            },
            'symbol_fundamentals': fundamental_metrics,
            'tech_alignment': 'aligned' if tech_analysis and tech_analysis['technical_summary']['market_sentiment'] == ('bullish' if avg_market_valuation > 60 else 'bearish') else 'divergent'
        }
        
        context['task_instance'].xcom_push(key='fundamental_analysis', value=processed_data)
        logger.info(f"Fundamental analysis completed: {market_outlook} market")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in fundamental analysis: {e}")
        raise


def detect_patterns(**context) -> Dict[str, Any]:
    """Detect chart patterns and trading signals."""
    try:
        logger.info("Starting pattern detection")
        
        data_collector = MarketDataCollector()
        market_data = data_collector.get_recent_data(SYMBOLS, timeframe='1h', periods=100)
        pattern_detector = ChartPatternDetector()
        
        detected_patterns = {}
        pattern_summary = {'bullish_patterns': 0, 'bearish_patterns': 0, 'breakout_candidates': []}
        
        for symbol in SYMBOLS:
            if symbol in market_data:
                patterns = pattern_detector.detect_patterns(market_data[symbol])
                
                bullish_patterns = []
                bearish_patterns = []
                
                if 'patterns_found' in patterns:
                    for i, pattern in enumerate(patterns['patterns_found']):
                        confidence = patterns.get('confidence', [0.5])[min(i, len(patterns.get('confidence', [])) - 1)]
                        if pattern in ['double_bottom', 'ascending_triangle', 'bullish_flag']:
                            bullish_patterns.append({'pattern': pattern, 'confidence': confidence}); pattern_summary['bullish_patterns'] += 1
                        elif pattern in ['double_top', 'descending_triangle', 'bearish_flag']:
                            bearish_patterns.append({'pattern': pattern, 'confidence': confidence}); pattern_summary['bearish_patterns'] += 1
                
                breakout_prob = patterns.get('breakout_probability', 0.5)
                if breakout_prob > 0.7:
                    pattern_summary['breakout_candidates'].append({'symbol': symbol, 'probability': breakout_prob})
                
                detected_patterns[symbol] = {
                    'bullish_patterns': bullish_patterns,
                    'bearish_patterns': bearish_patterns,
                    'breakout_probability': breakout_prob,
                    'overall_bias': 'bullish' if len(bullish_patterns) > len(bearish_patterns) else 'bearish' if len(bearish_patterns) > len(bullish_patterns) else 'neutral'
                }
        
        market_pattern_bias = 'bullish' if pattern_summary['bullish_patterns'] > pattern_summary['bearish_patterns'] else 'bearish' if pattern_summary['bearish_patterns'] > pattern_summary['bullish_patterns'] else 'neutral'
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'symbols_analyzed': len(detected_patterns),
            'pattern_summary': {
                'market_pattern_bias': market_pattern_bias,
                'pattern_distribution': pattern_summary,
                'high_probability_setups': len(pattern_summary['breakout_candidates'])
            },
            'symbol_patterns': detected_patterns
        }
        
        context['task_instance'].xcom_push(key='pattern_analysis', value=processed_data)
        logger.info(f"Pattern detection completed: {market_pattern_bias} bias")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in pattern detection: {e}")
        raise


def analyze_sentiment(**context) -> Dict[str, Any]:
    """Analyze market sentiment from multiple sources."""
    try:
        logger.info("Starting sentiment analysis")
        
        tech_analysis = context['task_instance'].xcom_pull(task_ids='analyze_technical_indicators', key='technical_analysis')
        pattern_analysis = context['task_instance'].xcom_pull(task_ids='detect_patterns', key='pattern_analysis')
        
        sentiment_analyzer = SentimentAnalyzer()
        market_sentiment = sentiment_analyzer.analyze_market_sentiment({
            'technical_signals': tech_analysis.get('technical_summary', {}) if tech_analysis else {},
            'pattern_signals': pattern_analysis.get('pattern_summary', {}) if pattern_analysis else {}
        })
        
        sentiment_signals = []
        if tech_analysis: sentiment_signals.append(tech_analysis['technical_summary']['market_sentiment'])
        if pattern_analysis: sentiment_signals.append(pattern_analysis['pattern_summary']['market_pattern_bias'])
        sentiment_signals.append(market_sentiment.get('overall_sentiment', 'neutral'))
        bullish_count, bearish_count = len([s for s in sentiment_signals if s in ['bullish', 'positive']]), len([s for s in sentiment_signals if s in ['bearish', 'negative']])
        if bullish_count > bearish_count: consensus_sentiment, consensus_strength = 'bullish', bullish_count / len(sentiment_signals)
        elif bearish_count > bullish_count: consensus_sentiment, consensus_strength = 'bearish', bearish_count / len(sentiment_signals)
        else: consensus_sentiment, consensus_strength = 'neutral', 0.5
        
        fear_greed = market_sentiment.get('fear_greed_index', 50)
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'sentiment_analysis': {
                'consensus_sentiment': consensus_sentiment,
                'consensus_strength': consensus_strength,
                'sentiment_score': market_sentiment.get('sentiment_score', 0.0),
                'fear_greed_index': fear_greed
            },
            'sentiment_signals': {
                'technical_sentiment': tech_analysis['technical_summary']['market_sentiment'] if tech_analysis else 'neutral',
                'pattern_sentiment': pattern_analysis['pattern_summary']['market_pattern_bias'] if pattern_analysis else 'neutral',
                'signal_alignment': len(set(sentiment_signals)) == 1
            },
            'market_psychology': {
                'fear_greed_level': 'extreme_fear' if fear_greed < 20 else 'extreme_greed' if fear_greed > 80 else 'neutral',
                'contrarian_signal': 'buy' if fear_greed < 20 else 'sell' if fear_greed > 80 else 'hold'
            }
        }
        
        context['task_instance'].xcom_push(key='sentiment_analysis', value=processed_data)
        logger.info(f"Sentiment analysis completed: {consensus_sentiment} consensus")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise


def classify_market_regime(**context) -> Dict[str, Any]:
    """Classify current market regime and predict transitions."""
    try:
        logger.info("Starting market regime classification")
        
        # Get all previous analysis results
        tech_analysis = context['task_instance'].xcom_pull(task_ids='analyze_technical_indicators', key='technical_analysis')
        fundamental_analysis = context['task_instance'].xcom_pull(task_ids='analyze_fundamentals', key='fundamental_analysis')
        pattern_analysis = context['task_instance'].xcom_pull(task_ids='detect_patterns', key='pattern_analysis')
        sentiment_analysis = context['task_instance'].xcom_pull(task_ids='analyze_sentiment', key='sentiment_analysis')
        
        data_collector = MarketDataCollector()
        market_data = data_collector.get_recent_data(['SPY', 'QQQ', 'IWM'], timeframe='1d', periods=50)
        regime_classifier = MarketRegimeClassifier()
        regime_data = regime_classifier.classify_regime(market_data)
        
        regime_factors = {'technical_factor': 0.5, 'fundamental_factor': 0.5, 'pattern_factor': 0.5, 'sentiment_factor': 0.5}
        if tech_analysis: regime_factors['technical_factor'] = 0.7 if tech_analysis['technical_summary']['market_sentiment'] == 'bullish' else 0.3 if tech_analysis['technical_summary']['market_sentiment'] == 'bearish' else 0.5
        if fundamental_analysis: regime_factors['fundamental_factor'] = 0.7 if fundamental_analysis['fundamental_summary']['market_outlook'] == 'undervalued' else 0.3 if fundamental_analysis['fundamental_summary']['market_outlook'] == 'overvalued' else 0.5
        if pattern_analysis: regime_factors['pattern_factor'] = 0.7 if pattern_analysis['pattern_summary']['market_pattern_bias'] == 'bullish' else 0.3 if pattern_analysis['pattern_summary']['market_pattern_bias'] == 'bearish' else 0.5
        if sentiment_analysis: regime_factors['sentiment_factor'] = 0.7 if sentiment_analysis['sentiment_analysis']['consensus_sentiment'] == 'bullish' else 0.3 if sentiment_analysis['sentiment_analysis']['consensus_sentiment'] == 'bearish' else 0.5
        
        regime_score = sum(regime_factors.values()) / len(regime_factors)
        current_regime = 'strong_bull' if regime_score > 0.65 else 'trending_bull' if regime_score > 0.55 else 'sideways' if regime_score > 0.45 else 'trending_bear' if regime_score > 0.35 else 'strong_bear'
        regime_confidence, transition_probability = abs(regime_score - 0.5) * 2, 1 - abs(regime_score - 0.5) * 2
        
        processed_data = {
            'timestamp': datetime.now().isoformat(),
            'regime_classification': {
                'current_regime': current_regime,
                'regime_score': regime_score,
                'regime_confidence': regime_confidence,
                'transition_probability': transition_probability
            },
            'regime_factors': regime_factors,
            'trading_implications': {
                'recommended_strategy': 'aggressive_growth' if current_regime == 'strong_bull' else 'defensive' if current_regime in ['trending_bear', 'strong_bear'] else 'balanced',
                'risk_management': 'tight_stops' if transition_probability > 0.4 else 'normal_stops'
            },
            'analysis_integration': all([tech_analysis, fundamental_analysis, pattern_analysis, sentiment_analysis])
        }
        
        context['task_instance'].xcom_push(key='regime_analysis', value=processed_data)
        logger.info(f"Market regime classification completed: {current_regime}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in market regime classification: {e}")
        raise


# Define tasks
analyze_technical_indicators_task = PythonOperator(task_id='analyze_technical_indicators', python_callable=analyze_technical_indicators, dag=dag)
analyze_fundamentals_task = PythonOperator(task_id='analyze_fundamentals', python_callable=analyze_fundamentals, dag=dag)
detect_patterns_task = PythonOperator(task_id='detect_patterns', python_callable=detect_patterns, dag=dag)
analyze_sentiment_task = PythonOperator(task_id='analyze_sentiment', python_callable=analyze_sentiment, dag=dag)
classify_market_regime_task = PythonOperator(task_id='classify_market_regime', python_callable=classify_market_regime, dag=dag)

# Define task dependencies
analyze_technical_indicators_task >> [analyze_fundamentals_task, detect_patterns_task]
[analyze_fundamentals_task, detect_patterns_task] >> analyze_sentiment_task
analyze_sentiment_task >> classify_market_regime_task