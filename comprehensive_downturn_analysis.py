#!/usr/bin/env python3
"""
Comprehensive Market Turn Detection Analysis
Tests multiple threshold/timeframe combinations with trend/momentum confirmations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

class ComprehensiveDownturnAnalyzer:
    """Comprehensive analysis of different downturn detection approaches"""
    
    def __init__(self, csv_file):
        """Load and prepare ASI data"""
        print("="*80)
        print("COMPREHENSIVE MARKET TURN DETECTION ANALYSIS")
        print("="*80)
        print("\nLoading data...")
        
        self.df = pd.read_csv(csv_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['close'] = pd.to_numeric(self.df['close'])
        self.df['returns'] = self.df['close'].pct_change()
        
        print(f"Loaded {len(self.df)} records from {self.df['date'].min()} to {self.df['date'].max()}")
    
    def compute_technical_indicators(self):
        """Compute comprehensive technical indicators"""
        print("\nComputing technical indicators...")
        
        # Moving Averages
        self.df['sma_20'] = self.df['close'].rolling(20).mean()
        self.df['sma_50'] = self.df['close'].rolling(50).mean()
        self.df['sma_200'] = self.df['close'].rolling(200).mean()
        self.df['ema_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']
        
        # RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = -delta.clip(upper=0).rolling(period).mean()
            rs = gain / (loss + 1e-9)
            return 100 - 100 / (1 + rs)
        
        self.df['rsi_14'] = calculate_rsi(self.df['close'], 14)
        self.df['rsi_7'] = calculate_rsi(self.df['close'], 7)
        
        # Volatility
        self.df['volatility_20'] = self.df['returns'].rolling(20).std() * np.sqrt(252)
        self.df['volatility_60'] = self.df['returns'].rolling(60).std() * np.sqrt(252)
        
        # Momentum
        self.df['momentum_10'] = self.df['close'].pct_change(10)
        self.df['momentum_20'] = self.df['close'].pct_change(20)
        self.df['momentum_50'] = self.df['close'].pct_change(50)
        
        # Trend indicators
        self.df['price_vs_sma20'] = (self.df['close'] - self.df['sma_20']) / self.df['sma_20']
        self.df['price_vs_sma50'] = (self.df['close'] - self.df['sma_50']) / self.df['sma_50']
        self.df['price_vs_sma200'] = (self.df['close'] - self.df['sma_200']) / self.df['sma_200']
        self.df['sma20_vs_sma50'] = (self.df['sma_20'] - self.df['sma_50']) / self.df['sma_50']
        self.df['sma50_vs_sma200'] = (self.df['sma_50'] - self.df['sma_200']) / self.df['sma_200']
        
        # Local drawdowns
        for w in [20, 60, 120]:
            peak_w = self.df['close'].rolling(w).max()
            self.df[f'local_dd_{w}'] = (self.df['close'] - peak_w) / peak_w
        
        # Returns
        self.df['return_1d'] = self.df['close'].pct_change(1)
        self.df['return_3d'] = self.df['close'].pct_change(3)
        self.df['return_5d'] = self.df['close'].pct_change(5)
        self.df['return_7d'] = self.df['close'].pct_change(7)
        self.df['return_10d'] = self.df['close'].pct_change(10)
        self.df['return_15d'] = self.df['close'].pct_change(15)
        
        # Trend break signals
        self.df['below_sma20'] = (self.df['close'] < self.df['sma_20']).astype(int)
        self.df['below_sma50'] = (self.df['close'] < self.df['sma_50']).astype(int)
        self.df['below_sma200'] = (self.df['close'] < self.df['sma_200']).astype(int)
        self.df['sma20_below_sma50'] = (self.df['sma_20'] < self.df['sma_50']).astype(int)
        self.df['sma50_below_sma200'] = (self.df['sma_50'] < self.df['sma_200']).astype(int)
        
        # Momentum signals
        self.df['negative_momentum_10'] = (self.df['momentum_10'] < 0).astype(int)
        self.df['negative_momentum_20'] = (self.df['momentum_20'] < 0).astype(int)
        self.df['negative_momentum_50'] = (self.df['momentum_50'] < 0).astype(int)
        
        # RSI signals
        self.df['rsi_below_50'] = (self.df['rsi_14'] < 50).astype(int)
        self.df['rsi_below_40'] = (self.df['rsi_14'] < 40).astype(int)
        self.df['rsi_declining'] = (self.df['rsi_14'] < self.df['rsi_14'].shift(1)).astype(int)
        
        # MACD signals
        self.df['macd_bearish'] = (self.df['macd'] < self.df['macd_signal']).astype(int)
        self.df['macd_hist_negative'] = (self.df['macd_hist'] < 0).astype(int)
        
        print("Technical indicators computed.")
    
    def create_labels(self, threshold, prediction_days):
        """Create labels for a specific threshold and timeframe"""
        label_col = f'will_have_drawdown_{abs(threshold)*100}pct_{prediction_days}d'
        self.df[label_col] = False
        self.df[f'max_dd_{abs(threshold)*100}pct_{prediction_days}d'] = 0.0
        
        for i in range(len(self.df) - prediction_days):
            current_price = self.df.loc[i, 'close']
            peak_price = current_price
            max_drawdown = 0.0
            
            lookahead_window = self.df.iloc[i+1:i+1+prediction_days]
            
            for j, row in lookahead_window.iterrows():
                price = row['close']
                if price > peak_price:
                    peak_price = price
                drawdown = (price - peak_price) / peak_price
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
            
            if max_drawdown <= threshold:
                self.df.loc[i, label_col] = True
            self.df.loc[i, f'max_dd_{abs(threshold)*100}pct_{prediction_days}d'] = max_drawdown
        
        return label_col
    
    def test_all_combinations(self):
        """Test all threshold and timeframe combinations"""
        print("\n" + "="*80)
        print("TESTING ALL COMBINATIONS")
        print("="*80)
        
        thresholds = [-0.03, -0.05, -0.07]  # -3%, -5%, -7%
        timeframes = [5, 7, 10, 15]  # days
        
        results = []
        
        for threshold in thresholds:
            for timeframe in timeframes:
                print(f"\nTesting: {threshold*100:.0f}% in {timeframe} days...")
                
                # Create labels
                label_col = self.create_labels(threshold, timeframe)
                
                # Filter to valid rows
                valid_mask = ~self.df[label_col].isna()
                valid_data = self.df[valid_mask].copy()
                
                if len(valid_data) == 0:
                    continue
                
                positive_cases = valid_data[label_col].sum()
                negative_cases = (valid_data[label_col] == False).sum()
                total_cases = len(valid_data)
                positive_rate = positive_cases / total_cases * 100
                
                # Calculate statistics
                avg_drawdown = valid_data[f'max_dd_{abs(threshold)*100}pct_{timeframe}d'].mean()
                
                # Check confirmations for positive cases
                positive_data = valid_data[valid_data[label_col] == True]
                
                if len(positive_data) > 0:
                    trend_break_rate = positive_data['below_sma50'].mean() * 100
                    negative_momentum_rate = positive_data['negative_momentum_20'].mean() * 100
                    rsi_below_50_rate = positive_data['rsi_below_50'].mean() * 100
                    macd_bearish_rate = positive_data['macd_bearish'].mean() * 100
                else:
                    trend_break_rate = 0
                    negative_momentum_rate = 0
                    rsi_below_50_rate = 0
                    macd_bearish_rate = 0
                
                results.append({
                    'threshold_pct': threshold * 100,
                    'timeframe_days': timeframe,
                    'total_cases': total_cases,
                    'positive_cases': positive_cases,
                    'negative_cases': negative_cases,
                    'positive_rate': positive_rate,
                    'avg_drawdown': avg_drawdown,
                    'trend_break_rate': trend_break_rate,
                    'negative_momentum_rate': negative_momentum_rate,
                    'rsi_below_50_rate': rsi_below_50_rate,
                    'macd_bearish_rate': macd_bearish_rate,
                    'label_col': label_col
                })
                
                print(f"  Positive cases: {positive_cases} ({positive_rate:.2f}%)")
                print(f"  Trend break rate: {trend_break_rate:.1f}%")
                print(f"  Negative momentum rate: {negative_momentum_rate:.1f}%")
        
        self.combination_results = pd.DataFrame(results)
        return self.combination_results
    
    def analyze_with_confirmations(self, threshold, timeframe):
        """Analyze a specific combination with confirmation signals"""
        label_col = f'will_have_drawdown_{abs(threshold)*100}pct_{timeframe}d'
        
        if label_col not in self.df.columns:
            self.create_labels(threshold, timeframe)
        
        valid_data = self.df[self.df[label_col].notna()].copy()
        positive_cases = valid_data[valid_data[label_col] == True]
        negative_cases = valid_data[valid_data[label_col] == False]
        
        # Analyze confirmations
        confirmations = {}
        
        for signal_name in ['below_sma50', 'negative_momentum_20', 'rsi_below_50', 
                           'macd_bearish', 'sma20_below_sma50']:
            if signal_name in positive_cases.columns:
                pos_rate = positive_cases[signal_name].mean() * 100
                neg_rate = negative_cases[signal_name].mean() * 100
                lift = pos_rate - neg_rate
                
                confirmations[signal_name] = {
                    'positive_rate': pos_rate,
                    'negative_rate': neg_rate,
                    'lift': lift
                }
        
        return confirmations
    
    def find_best_combinations(self):
        """Find best combinations based on multiple criteria"""
        print("\n" + "="*80)
        print("FINDING BEST COMBINATIONS")
        print("="*80)
        
        if not hasattr(self, 'combination_results'):
            print("Run test_all_combinations() first")
            return None
        
        df = self.combination_results.copy()
        
        # Score each combination
        # Higher score = better
        # Consider: reasonable positive rate (5-20%), good confirmations, actionable timeframe
        
        df['score'] = 0
        
        # Positive rate score (prefer 5-20%)
        df['pos_rate_score'] = np.where(
            (df['positive_rate'] >= 5) & (df['positive_rate'] <= 20),
            100 - abs(df['positive_rate'] - 12.5) * 4,  # Peak at 12.5%
            np.maximum(0, 100 - abs(df['positive_rate'] - 12.5) * 2)
        )
        
        # Confirmation score (higher is better)
        df['confirmation_score'] = (
            df['trend_break_rate'] * 0.3 +
            df['negative_momentum_rate'] * 0.3 +
            df['rsi_below_50_rate'] * 0.2 +
            df['macd_bearish_rate'] * 0.2
        )
        
        # Timeframe score (prefer 7-10 days for actionable lead time)
        df['timeframe_score'] = np.where(
            (df['timeframe_days'] >= 7) & (df['timeframe_days'] <= 10),
            100,
            np.maximum(0, 100 - abs(df['timeframe_days'] - 8.5) * 10)
        )
        
        # Threshold score (prefer -3% to -5% for early detection)
        df['threshold_score'] = np.where(
            (df['threshold_pct'] >= -5) & (df['threshold_pct'] <= -3),
            100,
            np.maximum(0, 100 - abs(df['threshold_pct'] + 4) * 20)
        )
        
        # Combined score
        df['total_score'] = (
            df['pos_rate_score'] * 0.25 +
            df['confirmation_score'] * 0.35 +
            df['timeframe_score'] * 0.20 +
            df['threshold_score'] * 0.20
        )
        
        # Sort by total score
        df = df.sort_values('total_score', ascending=False)
        
        print("\nTop 5 Combinations:")
        print("-"*80)
        for idx, row in df.head(5).iterrows():
            print(f"\n{idx+1}. {row['threshold_pct']:.0f}% in {row['timeframe_days']} days")
            print(f"   Score: {row['total_score']:.1f}")
            print(f"   Positive rate: {row['positive_rate']:.2f}%")
            print(f"   Confirmations: Trend={row['trend_break_rate']:.1f}%, "
                  f"Momentum={row['negative_momentum_rate']:.1f}%, "
                  f"RSI={row['rsi_below_50_rate']:.1f}%, "
                  f"MACD={row['macd_bearish_rate']:.1f}%")
        
        self.ranked_combinations = df
        return df
    
    def plot_comparison(self):
        """Plot comparison of all combinations"""
        if not hasattr(self, 'combination_results'):
            print("Run test_all_combinations() first")
            return
        
        df = self.combination_results
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Positive Rate by Combination
        pivot_pos_rate = df.pivot(index='timeframe_days', columns='threshold_pct', values='positive_rate')
        sns.heatmap(pivot_pos_rate, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 0])
        axes[0, 0].set_title('Positive Rate (%) by Threshold and Timeframe')
        axes[0, 0].set_xlabel('Threshold (%)')
        axes[0, 0].set_ylabel('Timeframe (days)')
        
        # 2. Confirmation Rates
        pivot_trend = df.pivot(index='timeframe_days', columns='threshold_pct', values='trend_break_rate')
        sns.heatmap(pivot_trend, annot=True, fmt='.1f', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('Trend Break Rate (%) - Price Below SMA50')
        axes[0, 1].set_xlabel('Threshold (%)')
        axes[0, 1].set_ylabel('Timeframe (days)')
        
        # 3. Negative Momentum Rate
        pivot_momentum = df.pivot(index='timeframe_days', columns='threshold_pct', values='negative_momentum_rate')
        sns.heatmap(pivot_momentum, annot=True, fmt='.1f', cmap='Greens', ax=axes[1, 0])
        axes[1, 0].set_title('Negative Momentum Rate (%) - 20-day < 0')
        axes[1, 0].set_xlabel('Threshold (%)')
        axes[1, 0].set_ylabel('Timeframe (days)')
        
        # 4. Total Score (if ranked)
        if hasattr(self, 'ranked_combinations'):
            pivot_score = self.ranked_combinations.pivot(
                index='timeframe_days', columns='threshold_pct', values='total_score'
            )
            sns.heatmap(pivot_score, annot=True, fmt='.1f', cmap='viridis', ax=axes[1, 1])
            axes[1, 1].set_title('Total Score (Higher = Better)')
            axes[1, 1].set_xlabel('Threshold (%)')
            axes[1, 1].set_ylabel('Timeframe (days)')
        
        plt.tight_layout()
        plt.savefig('combination_comparison.png', dpi=150)
        print("\nComparison plots saved to combination_comparison.png")
        plt.show()
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        
        if not hasattr(self, 'ranked_combinations'):
            print("Run find_best_combinations() first")
            return
        
        best = self.ranked_combinations.iloc[0]
        
        print(f"\n🏆 BEST COMBINATION:")
        print(f"   {best['threshold_pct']:.0f}% drawdown in {best['timeframe_days']} days")
        print(f"   Score: {best['total_score']:.1f}/100")
        print(f"   Positive rate: {best['positive_rate']:.2f}%")
        
        print(f"\n📊 CONFIRMATION SIGNALS:")
        print(f"   Trend break (below SMA50): {best['trend_break_rate']:.1f}%")
        print(f"   Negative momentum: {best['negative_momentum_rate']:.1f}%")
        print(f"   RSI below 50: {best['rsi_below_50_rate']:.1f}%")
        print(f"   MACD bearish: {best['macd_bearish_rate']:.1f}%")
        
        print(f"\n💡 RECOMMENDED APPROACH:")
        print(f"   1. Primary Signal: {best['threshold_pct']:.0f}% drawdown in {best['timeframe_days']} days")
        print(f"   2. Confirm with:")
        print(f"      - Price below SMA50 ({best['trend_break_rate']:.0f}% of cases)")
        print(f"      - Negative 20-day momentum ({best['negative_momentum_rate']:.0f}% of cases)")
        print(f"      - RSI below 50 ({best['rsi_below_50_rate']:.0f}% of cases)")
        
        # Multi-tier recommendation
        print(f"\n🎯 MULTI-TIER WARNING SYSTEM:")
        
        # Find early warning (lower threshold)
        early = self.ranked_combinations[
            (self.ranked_combinations['threshold_pct'] >= -4) & 
            (self.ranked_combinations['timeframe_days'] >= 7)
        ].iloc[0] if len(self.ranked_combinations[
            (self.ranked_combinations['threshold_pct'] >= -4) & 
            (self.ranked_combinations['timeframe_days'] >= 7)
        ]) > 0 else None
        
        if early is not None:
            print(f"   ⚠️  Early Warning: {early['threshold_pct']:.0f}% in {early['timeframe_days']} days")
        
        print(f"   🚨 Confirmed Turn: {best['threshold_pct']:.0f}% in {best['timeframe_days']} days")
        
        # Find major downturn
        major = self.ranked_combinations[
            (self.ranked_combinations['threshold_pct'] <= -6) & 
            (self.ranked_combinations['timeframe_days'] >= 10)
        ].iloc[0] if len(self.ranked_combinations[
            (self.ranked_combinations['threshold_pct'] <= -6) & 
            (self.ranked_combinations['timeframe_days'] >= 10)
        ]) > 0 else None
        
        if major is not None:
            print(f"   🔴 Major Downturn: {major['threshold_pct']:.0f}% in {major['timeframe_days']} days")
        
        return best
    
    def save_results(self, output_file='comprehensive_analysis_results.json'):
        """Save all results"""
        results = {
            'analysis_date': datetime.now().isoformat(),
            'total_records': len(self.df),
            'date_range': {
                'start': self.df['date'].min().isoformat(),
                'end': self.df['date'].max().isoformat()
            },
            'combinations_tested': self.combination_results.to_dict('records') if hasattr(self, 'combination_results') else None,
            'best_combination': self.ranked_combinations.iloc[0].to_dict() if hasattr(self, 'ranked_combinations') else None,
            'top_5_combinations': self.ranked_combinations.head(5).to_dict('records') if hasattr(self, 'ranked_combinations') else None
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Also save CSV
        if hasattr(self, 'ranked_combinations'):
            self.ranked_combinations.to_csv('combination_rankings.csv', index=False)
            print(f"\nResults saved to {output_file} and combination_rankings.csv")


def main():
    """Main analysis function"""
    csv_file = 'ASI_close_prices_last_15_years_2025-12-05.csv'
    
    # Initialize analyzer
    analyzer = ComprehensiveDownturnAnalyzer(csv_file)
    
    # Compute indicators
    analyzer.compute_technical_indicators()
    
    # Test all combinations
    analyzer.test_all_combinations()
    
    # Find best combinations
    analyzer.find_best_combinations()
    
    # Generate recommendations
    best = analyzer.generate_recommendations()
    
    # Plot comparison
    analyzer.plot_comparison()
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext step: Train models on the best combination!")


if __name__ == "__main__":
    main()

