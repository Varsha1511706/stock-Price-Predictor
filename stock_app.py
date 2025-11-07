import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Stock Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .buy-signal {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
    .sell-signal {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 0.5rem 0;
    }
    .hold-signal {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 0.5rem 0;
    }
    .strong-buy {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    .strong-sell {
        background: linear-gradient(135deg, #dc3545, #e83e8c);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class MoneyControlAnalyzer:
    def __init__(self):
        self.stock_data = None
        self.technical_data = None

    def get_stock_data(self, symbol, period="6mo"):
        """Get stock data from Yahoo Finance"""
        try:
            with st.spinner(f'📊 Fetching data for {symbol}...'):
                stock = yf.download(symbol, period=period, progress=False)
                if stock.empty:
                    st.error(f"❌ No data found for {symbol}")
                    return None

                # ✅ FIX: Handle MultiIndex columns properly
                if isinstance(stock.columns, pd.MultiIndex):
                    # Extract just the first level (price type) and ignore the symbol
                    stock.columns = stock.columns.get_level_values(0)
                
                # If columns still have symbols appended, clean them
                cleaned_columns = {}
                for col in stock.columns:
                    if isinstance(col, str) and '_' in col:
                        # Extract just the price type (Open, High, Low, Close, Volume)
                        clean_col = col.split('_')[0]
                        cleaned_columns[col] = clean_col
                    else:
                        cleaned_columns[col] = col
                
                stock = stock.rename(columns=cleaned_columns)
                
                # Ensure we have the required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in stock.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {missing_columns}")
                    st.info(f"📋 Available columns: {list(stock.columns)}")
                    return None

                self.stock_data = stock
                st.success(f"✅ Successfully fetched {len(stock)} days of data")
                st.info(f"📊 Columns available: {list(stock.columns)}")
                return stock
                
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            return None

    def calculate_technical_indicators(self):
        """Calculate advanced technical indicators"""
        if self.stock_data is None:
            st.error("❌ No stock data available")
            return None

        try:
            data = self.stock_data.copy()
            
            # ✅ Debug: Show available columns
            st.info(f"📋 Calculating indicators with columns: {list(data.columns)}")
            
            # Ensure we have the required columns
            if 'Close' not in data.columns:
                st.error("❌ 'Close' column not found in data")
                return None

            # RSI Calculation
            st.info("📈 Calculating RSI...")
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # Moving Averages
            st.info("📈 Calculating Moving Averages...")
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()

            # MACD
            st.info("📈 Calculating MACD...")
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

            # Bollinger Bands
            st.info("📈 Calculating Bollinger Bands...")
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

            # Stochastic Oscillator
            st.info("📈 Calculating Stochastic...")
            low_14 = data['Low'].rolling(window=14).min()
            high_14 = data['High'].rolling(window=14).max()
            data['Stoch_K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
            data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

            # Volume indicators
            st.info("📈 Calculating Volume indicators...")
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']

            # Price Momentum
            data['Momentum_5'] = data['Close'].pct_change(periods=5)
            data['Momentum_10'] = data['Close'].pct_change(periods=10)

            # Volatility
            data['Volatility'] = data['Close'].pct_change().rolling(window=10).std()

            # Drop NaN values
            initial_count = len(data)
            data = data.dropna()
            final_count = len(data)
            
            st.info(f"✅ Removed {initial_count - final_count} rows with NaN values")

            self.technical_data = data
            st.success("🎉 All technical indicators calculated successfully!")
            return data
            
        except Exception as e:
            st.error(f"❌ Error calculating indicators: {e}")
            return None

    def generate_signals(self):
        """Generate buy/sell signals based on multiple indicators"""
        if self.technical_data is None:
            st.error("❌ No technical data available")
            return None

        try:
            # Get the latest row safely
            current = self.technical_data.iloc[-1]
            
            # ✅ Convert to float to avoid Series comparison issues
            current_rsi = float(current['RSI'])
            current_close = float(current['Close'])
            current_sma20 = float(current['SMA_20'])
            current_sma50 = float(current['SMA_50'])
            current_macd = float(current['MACD'])
            current_macd_signal = float(current['MACD_Signal'])
            current_bb_upper = float(current['BB_Upper'])
            current_bb_lower = float(current['BB_Lower'])
            current_stoch_k = float(current['Stoch_K'])
            current_stoch_d = float(current['Stoch_D'])
            current_volume_ratio = float(current['Volume_Ratio'])

            signals = []
            score = 0

            # RSI Signal
            if current_rsi < 30:
                signals.append(("RSI", "BUY", f"Oversold (RSI: {current_rsi:.1f})"))
                score += 2
            elif current_rsi > 70:
                signals.append(("RSI", "SELL", f"Overbought (RSI: {current_rsi:.1f})"))
                score -= 2
            else:
                signals.append(("RSI", "HOLD", f"Neutral (RSI: {current_rsi:.1f})"))

            # Moving Average Signal
            if current_sma20 > current_sma50:
                signals.append(("Moving Average", "BUY", "Bullish crossover (SMA20 > SMA50)"))
                score += 1
            else:
                signals.append(("Moving Average", "SELL", "Bearish crossover (SMA20 < SMA50)"))
                score -= 1

            # MACD Signal
            if current_macd > current_macd_signal:
                signals.append(("MACD", "BUY", "Bullish momentum"))
                score += 1
            else:
                signals.append(("MACD", "SELL", "Bearish momentum"))
                score -= 1

            # Bollinger Bands Signal
            if current_close < current_bb_lower:
                signals.append(("Bollinger Bands", "BUY", "Oversold - Near lower band"))
                score += 2
            elif current_close > current_bb_upper:
                signals.append(("Bollinger Bands", "SELL", "Overbought - Near upper band"))
                score -= 2
            else:
                signals.append(("Bollinger Bands", "HOLD", "Within normal range"))

            # Stochastic Signal
            if current_stoch_k < 20 and current_stoch_d < 20:
                signals.append(("Stochastic", "BUY", f"Oversold (K: {current_stoch_k:.1f}, D: {current_stoch_d:.1f})"))
                score += 1
            elif current_stoch_k > 80 and current_stoch_d > 80:
                signals.append(("Stochastic", "SELL", f"Overbought (K: {current_stoch_k:.1f}, D: {current_stoch_d:.1f})"))
                score -= 1
            else:
                signals.append(("Stochastic", "HOLD", f"Neutral (K: {current_stoch_k:.1f}, D: {current_stoch_d:.1f})"))

            # Volume Signal
            if current_volume_ratio > 1.5:
                signals.append(("Volume", "BUY", f"High volume (Ratio: {current_volume_ratio:.2f})"))
                score += 1
            elif current_volume_ratio < 0.5:
                signals.append(("Volume", "SELL", f"Low volume (Ratio: {current_volume_ratio:.2f})"))
                score -= 1
            else:
                signals.append(("Volume", "HOLD", f"Normal volume (Ratio: {current_volume_ratio:.2f})"))

            # Determine overall signal
            if score >= 4:
                overall_signal = "STRONG BUY 🚀"
                signal_class = "strong-buy"
            elif score >= 2:
                overall_signal = "BUY 📈"
                signal_class = "buy-signal"
            elif score <= -4:
                overall_signal = "STRONG SELL 🔻"
                signal_class = "strong-sell"
            elif score <= -2:
                overall_signal = "SELL 📉"
                signal_class = "sell-signal"
            else:
                overall_signal = "HOLD ⚖️"
                signal_class = "hold-signal"

            return {
                'signals': signals,
                'overall_signal': overall_signal,
                'signal_class': signal_class,
                'score': score,
                'current_data': current
            }
            
        except Exception as e:
            st.error(f"❌ Error generating signals: {e}")
            return None

    def calculate_support_resistance(self):
        """Calculate support and resistance levels"""
        if self.technical_data is None:
            return None

        try:
            data = self.technical_data
            
            # Use last 20 days for calculations
            recent_data = data.tail(20)
            recent_high = float(recent_data['High'].max())
            recent_low = float(recent_data['Low'].min())
            current_price = float(data['Close'].iloc[-1])

            # Pivot point calculations
            pivot = (recent_high + recent_low + current_price) / 3
            resistance1 = (2 * pivot) - recent_low
            support1 = (2 * pivot) - recent_high
            resistance2 = pivot + (recent_high - recent_low)
            support2 = pivot - (recent_high - recent_low)

            return {
                'support_levels': [support1, support2],
                'resistance_levels': [resistance1, resistance2],
                'pivot': pivot,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'current_price': current_price
            }
        except Exception as e:
            st.error(f"❌ Error calculating support/resistance: {e}")
            return None


def main():
    st.markdown('<h1 class="main-header">💰 Advanced Stock Analyzer</h1>', unsafe_allow_html=True)

    analyzer = MoneyControlAnalyzer()

    with st.sidebar:
        st.header("🎯 Stock Selection")

        stock_options = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services",
            "INFY.NS": "Infosys",
            "HDFCBANK.NS": "HDFC Bank",
            "IDEA.NS": "Vodafone Idea",
            "ITC.NS": "ITC Limited",
            "SBIN.NS": "State Bank of India",
            "WIPRO.NS": "Wipro",
            "ONGC.NS": "ONGC",
            "HINDUNILVR.NS": "Hindustan Unilever",
            "AAPL": "Apple Inc",
            "TSLA": "Tesla Inc"
        }

        selected_stock = st.selectbox(
            "Select Stock:",
            options=list(stock_options.keys()),
            format_func=lambda x: stock_options[x]
        )

        period = st.selectbox(
            "Data Period:",
            ["3mo", "6mo", "1y", "2y"],
            index=1
        )

        analyze_btn = st.button("🚀 Analyze Stock", type="primary", use_container_width=True)

    # Quick analysis buttons
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Analysis")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("RELIANCE", use_container_width=True):
            selected_stock = "RELIANCE.NS"
            analyze_btn = True
        if st.button("TCS", use_container_width=True):
            selected_stock = "TCS.NS"
            analyze_btn = True
    with col2:
        if st.button("IDEA", use_container_width=True):
            selected_stock = "IDEA.NS"
            analyze_btn = True
        if st.button("AAPL", use_container_width=True):
            selected_stock = "AAPL"
            analyze_btn = True

    if analyze_btn:
        with st.spinner("🔬 Fetching stock data and analyzing..."):
            # Get stock data
            stock_data = analyzer.get_stock_data(selected_stock, period)

            if stock_data is not None:
                # Calculate technical indicators
                analyzer.calculate_technical_indicators()
                
                if analyzer.technical_data is not None:
                    # Generate signals
                    signal_data = analyzer.generate_signals()
                    levels = analyzer.calculate_support_resistance()

                    if signal_data and levels:
                        # Display main signal
                        st.markdown(f"""
                        <div class="{signal_data['signal_class']}">
                            <h2 style="margin:0; font-size: 2.5rem;">{signal_data['overall_signal']}</h2>
                            <h3 style="margin:0;">Confidence Score: {signal_data['score']}/8</h3>
                            <p style="margin:0;">Based on comprehensive technical analysis</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Current metrics
                        current = signal_data['current_data']
                        st.subheader("📊 Current Market Data")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Current Price", f"₹{float(current['Close']):.2f}")
                        with col2:
                            price_change = (float(current['Close']) - float(analyzer.technical_data['Close'].iloc[-2])) / float(analyzer.technical_data['Close'].iloc[-2]) * 100
                            st.metric("Daily Change", f"{price_change:+.2f}%")
                        with col3:
                            st.metric("RSI", f"{float(current['RSI']):.1f}")
                        with col4:
                            st.metric("Volume Ratio", f"{float(current['Volume_Ratio']):.2f}x")
                        with col5:
                            st.metric("Volatility", f"{(float(current['Volatility']) * 100):.2f}%")

                        # Technical signals
                        st.subheader("📈 Technical Indicator Signals")
                        
                        for indicator, signal, reason in signal_data['signals']:
                            if signal == "BUY":
                                st.markdown(f"<div class='buy-signal'><strong>✅ {indicator}:</strong> {signal} - {reason}</div>", unsafe_allow_html=True)
                            elif signal == "SELL":
                                st.markdown(f"<div class='sell-signal'><strong>❌ {indicator}:</strong> {signal} - {reason}</div>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div class='hold-signal'><strong>⚖️ {indicator}:</strong> {signal} - {reason}</div>", unsafe_allow_html=True)

                        # Support & Resistance
                        st.subheader("🎯 Key Price Levels")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**🛡️ Support Levels:**")
                            for i, level in enumerate(levels['support_levels'], 1):
                                distance_pct = ((levels['current_price'] - level) / levels['current_price']) * 100
                                st.write(f"S{i}: ₹{level:.2f} ({distance_pct:+.1f}%)")
                        
                        with col2:
                            st.write("**🎯 Resistance Levels:**")
                            for i, level in enumerate(levels['resistance_levels'], 1):
                                distance_pct = ((level - levels['current_price']) / levels['current_price']) * 100
                                st.write(f"R{i}: ₹{level:.2f} ({distance_pct:+.1f}%)")

                        # Charts
                        st.subheader("📊 Technical Charts")
                        
                        tab1, tab2, tab3 = st.tabs(["Price & Trends", "Momentum Indicators", "Volume Analysis"])
                        
                        with tab1:
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                            
                            # Price with moving averages
                            ax1.plot(analyzer.technical_data.index, analyzer.technical_data['Close'], label='Price', linewidth=2, color='blue')
                            ax1.plot(analyzer.technical_data.index, analyzer.technical_data['SMA_20'], label='SMA 20', linestyle='--', alpha=0.8)
                            ax1.plot(analyzer.technical_data.index, analyzer.technical_data['SMA_50'], label='SMA 50', linestyle='--', alpha=0.8)
                            ax1.set_title('Price & Moving Averages')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # Bollinger Bands
                            ax2.plot(analyzer.technical_data.index, analyzer.technical_data['Close'], label='Price', linewidth=1)
                            ax2.plot(analyzer.technical_data.index, analyzer.technical_data['BB_Upper'], label='Upper Band', linestyle='--')
                            ax2.plot(analyzer.technical_data.index, analyzer.technical_data['BB_Middle'], label='Middle Band', linestyle='--')
                            ax2.plot(analyzer.technical_data.index, analyzer.technical_data['BB_Lower'], label='Lower Band', linestyle='--')
                            ax2.set_title('Bollinger Bands')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with tab2:
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                            
                            # RSI
                            ax1.plot(analyzer.technical_data.index, analyzer.technical_data['RSI'], label='RSI', linewidth=2, color='purple')
                            ax1.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
                            ax1.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
                            ax1.set_title('RSI Indicator')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # MACD
                            ax2.plot(analyzer.technical_data.index, analyzer.technical_data['MACD'], label='MACD', linewidth=2)
                            ax2.plot(analyzer.technical_data.index, analyzer.technical_data['MACD_Signal'], label='Signal', linewidth=2)
                            ax2.bar(analyzer.technical_data.index, analyzer.technical_data['MACD_Histogram'], alpha=0.3, color='gray')
                            ax2.set_title('MACD')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with tab3:
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                            
                            # Volume
                            ax1.bar(analyzer.technical_data.index, analyzer.technical_data['Volume'], alpha=0.7, color='orange')
                            ax1.plot(analyzer.technical_data.index, analyzer.technical_data['Volume_SMA'], color='red', linewidth=2)
                            ax1.set_title('Trading Volume')
                            ax1.grid(True, alpha=0.3)
                            
                            # Stochastic
                            ax2.plot(analyzer.technical_data.index, analyzer.technical_data['Stoch_K'], label='%K', linewidth=2)
                            ax2.plot(analyzer.technical_data.index, analyzer.technical_data['Stoch_D'], label='%D', linewidth=2)
                            ax2.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Overbought')
                            ax2.axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Oversold')
                            ax2.set_title('Stochastic Oscillator')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)

                        # Trading Recommendation
                        st.subheader("💡 Trading Recommendation")
                        
                        if signal_data['score'] >= 3:
                            st.success("""
                            **🎯 BULLISH STRATEGY:**
                            - Consider entering long positions
                            - Set stop loss below key support levels
                            - Target resistance levels for profit taking
                            - Monitor for trend continuation signals
                            """)
                        elif signal_data['score'] <= -3:
                            st.error("""
                            **🔻 BEARISH STRATEGY:**
                            - Consider short positions or exit longs
                            - Set stop loss above key resistance levels
                            - Target support levels for covering shorts
                            - Wait for trend reversal confirmation
                            """)
                        else:
                            st.info("""
                            **⚖️ NEUTRAL STRATEGY:**
                            - Wait for clearer market direction
                            - Monitor key support/resistance levels
                            - Consider range-bound trading strategies
                            - Prepare for potential breakout
                            """)

        # Risk Disclaimer
        st.markdown("---")
        st.error("""
        **⚠️ RISK DISCLAIMER:** 
        This analysis is for EDUCATIONAL PURPOSES only. Not financial advice. 
        Past performance doesn't guarantee future results. Always consult qualified 
        financial advisors before making investment decisions. Stock market 
        investments are subject to market risks.
        """)


if __name__ == "__main__":
    main()