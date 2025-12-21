#!/usr/bin/env python3
"""
Analyze Drawdown Depth and Duration
For -5% drawdowns in 10 days, calculate:
- Average maximum drawdown (how far it drops)
- Average duration (how long it lasts)
- Standard deviations for both
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DrawdownDepthDurationAnalyzer:
    """Analyze depth and duration of -5% drawdowns in 10 days"""
    
    def __init__(self, csv_file):
        """Load and prepare ASI data"""
        print("="*80)
        print("DRAWDOWN DEPTH & DURATION ANALYSIS")
        print("Analyzing -5% drawdowns in 10 days")
        print("="*80)
        print("\nLoading data...")
        
        self.df = pd.read_csv(csv_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.df['close'] = pd.to_numeric(self.df['close'])
        
        print(f"Loaded {len(self.df)} records from {self.df['date'].min()} to {self.df['date'].max()}")
    
    def identify_drawdowns(self, threshold=-0.05, lookahead_days=10):
        """Identify all -5% drawdowns and track their full depth and duration"""
        print(f"\nIdentifying -{abs(threshold)*100}% drawdowns in {lookahead_days} days...")
        
        drawdowns = []
        processed_indices = set()
        
        for i in range(len(self.df) - lookahead_days):
            if i in processed_indices:
                continue
                
            # Start from current day
            start_idx = i
            start_price = self.df.loc[i, 'close']
            start_date = self.df.loc[i, 'date']
            peak_price = start_price
            peak_idx = i
            
            # Track drawdown within lookahead window
            max_drawdown_in_window = 0.0
            trough_idx_in_window = None
            
            # Look ahead for drawdown
            lookahead_window = self.df.iloc[i+1:i+1+lookahead_days]
            
            for j, row in lookahead_window.iterrows():
                price = row['close']
                
                # Update peak
                if price > peak_price:
                    peak_price = price
                    peak_idx = j
                
                # Calculate drawdown from peak
                drawdown = (price - peak_price) / peak_price
                
                if drawdown < max_drawdown_in_window:
                    max_drawdown_in_window = drawdown
                    trough_idx_in_window = j
            
            # Check if threshold was breached
            if max_drawdown_in_window <= threshold:
                # Found a drawdown! Now track full depth and duration
                peak_price = self.df.loc[peak_idx, 'close']
                peak_date = self.df.loc[peak_idx, 'date']
                
                # Find the actual trough (lowest point from peak)
                trough_idx = trough_idx_in_window
                trough_price = self.df.loc[trough_idx, 'close']
                trough_date = self.df.loc[trough_idx, 'date']
                max_drawdown = max_drawdown_in_window
                
                # Continue tracking beyond lookahead window to find full extent
                # Look further ahead to find if it drops more
                extended_window = self.df.iloc[trough_idx+1:min(trough_idx+1+60, len(self.df))]  # Look 60 days ahead
                
                for k, row in extended_window.iterrows():
                    price = row['close']
                    drawdown_from_peak = (price - peak_price) / peak_price
                    
                    if drawdown_from_peak < max_drawdown:
                        max_drawdown = drawdown_from_peak
                        trough_idx = k
                        trough_price = price
                        trough_date = row['date']
                
                # Find recovery point (back to -4% from peak, or new peak)
                recovery_idx = None
                recovery_date = None
                recovery_price = None
                
                # Look ahead from trough for recovery
                recovery_window = self.df.iloc[trough_idx+1:min(trough_idx+1+120, len(self.df))]  # Look 120 days ahead
                
                for k, row in recovery_window.iterrows():
                    price = row['close']
                    
                    # Recovery: price back to within 1% of peak, or new peak
                    recovery_threshold = peak_price * 0.99  # Within 1% of peak
                    
                    if price >= recovery_threshold or price >= peak_price:
                        recovery_idx = k
                        recovery_date = row['date']
                        recovery_price = price
                        break
                
                # Calculate durations
                days_to_trough = (trough_date - peak_date).days
                days_to_recovery = (recovery_date - peak_date).days if recovery_date else None
                total_duration = days_to_recovery if recovery_date else None
                
                drawdowns.append({
                    'start_date': start_date,
                    'peak_date': peak_date,
                    'peak_price': peak_price,
                    'trough_date': trough_date,
                    'trough_price': trough_price,
                    'max_drawdown': max_drawdown,
                    'recovery_date': recovery_date,
                    'recovery_price': recovery_price,
                    'days_to_trough': days_to_trough,
                    'days_to_recovery': days_to_recovery,
                    'total_duration': total_duration,
                    'peak_idx': peak_idx,
                    'trough_idx': trough_idx,
                    'recovery_idx': recovery_idx
                })
                
                # Mark indices as processed to avoid double counting
                if recovery_idx:
                    for idx in range(peak_idx, recovery_idx + 1):
                        processed_indices.add(idx)
                else:
                    for idx in range(peak_idx, min(trough_idx + 60, len(self.df))):
                        processed_indices.add(idx)
        
        self.drawdowns_df = pd.DataFrame(drawdowns)
        print(f"Found {len(self.drawdowns_df)} drawdowns")
        
        return self.drawdowns_df
    
    def analyze_statistics(self):
        """Calculate statistics on drawdown depth and duration"""
        if not hasattr(self, 'drawdowns_df') or len(self.drawdowns_df) == 0:
            print("No drawdowns found. Run identify_drawdowns() first.")
            return None
        
        df = self.drawdowns_df
        
        print("\n" + "="*80)
        print("DRAWDOWN STATISTICS")
        print("="*80)
        
        # Convert drawdown to percentage
        df['max_drawdown_pct'] = df['max_drawdown'] * 100
        
        # Statistics for maximum drawdown depth
        print("\n📉 MAXIMUM DRAWDOWN DEPTH (from peak):")
        print("-"*80)
        avg_depth = df['max_drawdown_pct'].mean()
        std_depth = df['max_drawdown_pct'].std()
        median_depth = df['max_drawdown_pct'].median()
        min_depth = df['max_drawdown_pct'].min()
        max_depth = df['max_drawdown_pct'].max()
        
        print(f"Average:     {avg_depth:.2f}%")
        print(f"Std Dev:     {std_depth:.2f}%")
        print(f"Median:      {median_depth:.2f}%")
        print(f"Minimum:     {min_depth:.2f}%")
        print(f"Maximum:     {max_depth:.2f}%")
        print(f"\nRange:       {avg_depth - std_depth:.2f}% to {avg_depth + std_depth:.2f}% (±1 std dev)")
        print(f"Range (95%): {avg_depth - 2*std_depth:.2f}% to {avg_depth + 2*std_depth:.2f}% (±2 std dev)")
        
        # Statistics for days to trough
        print("\n⏱️  DAYS TO TROUGH (from peak to lowest point):")
        print("-"*80)
        avg_days_to_trough = df['days_to_trough'].mean()
        std_days_to_trough = df['days_to_trough'].std()
        median_days_to_trough = df['days_to_trough'].median()
        min_days_to_trough = df['days_to_trough'].min()
        max_days_to_trough = df['days_to_trough'].max()
        
        print(f"Average:     {avg_days_to_trough:.1f} days")
        print(f"Std Dev:     {std_days_to_trough:.1f} days")
        print(f"Median:      {median_days_to_trough:.1f} days")
        print(f"Minimum:     {min_days_to_trough:.0f} days")
        print(f"Maximum:     {max_days_to_trough:.0f} days")
        print(f"\nRange:       {avg_days_to_trough - std_days_to_trough:.1f} to {avg_days_to_trough + std_days_to_trough:.1f} days (±1 std dev)")
        print(f"Range (95%): {avg_days_to_trough - 2*std_days_to_trough:.1f} to {avg_days_to_trough + 2*std_days_to_trough:.1f} days (±2 std dev)")
        
        # Statistics for total duration (if recovery found)
        recovered = df[df['total_duration'].notna()]
        
        if len(recovered) > 0:
            print("\n🔄 TOTAL DURATION (from peak to recovery):")
            print("-"*80)
            avg_duration = recovered['total_duration'].mean()
            std_duration = recovered['total_duration'].std()
            median_duration = recovered['total_duration'].median()
            min_duration = recovered['total_duration'].min()
            max_duration = recovered['total_duration'].max()
            
            print(f"Recovered cases: {len(recovered)} out of {len(df)} ({len(recovered)/len(df)*100:.1f}%)")
            print(f"Average:     {avg_duration:.1f} days")
            print(f"Std Dev:     {std_duration:.1f} days")
            print(f"Median:      {median_duration:.1f} days")
            print(f"Minimum:     {min_duration:.0f} days")
            print(f"Maximum:     {max_duration:.0f} days")
            print(f"\nRange:       {avg_duration - std_duration:.1f} to {avg_duration + std_duration:.1f} days (±1 std dev)")
            print(f"Range (95%): {avg_duration - 2*std_duration:.1f} to {avg_duration + 2*std_duration:.1f} days (±2 std dev)")
        
        # Summary statistics
        stats = {
            'total_drawdowns': len(df),
            'max_drawdown_depth': {
                'mean': float(avg_depth),
                'std': float(std_depth),
                'median': float(median_depth),
                'min': float(min_depth),
                'max': float(max_depth),
                'range_1std': [float(avg_depth - std_depth), float(avg_depth + std_depth)],
                'range_2std': [float(avg_depth - 2*std_depth), float(avg_depth + 2*std_depth)]
            },
            'days_to_trough': {
                'mean': float(avg_days_to_trough),
                'std': float(std_days_to_trough),
                'median': float(median_days_to_trough),
                'min': float(min_days_to_trough),
                'max': float(max_days_to_trough),
                'range_1std': [float(avg_days_to_trough - std_days_to_trough), float(avg_days_to_trough + std_days_to_trough)],
                'range_2std': [float(avg_days_to_trough - 2*std_days_to_trough), float(avg_days_to_trough + 2*std_days_to_trough)]
            }
        }
        
        if len(recovered) > 0:
            stats['total_duration'] = {
                'recovered_count': len(recovered),
                'recovery_rate': float(len(recovered) / len(df) * 100),
                'mean': float(avg_duration),
                'std': float(std_duration),
                'median': float(median_duration),
                'min': float(min_duration),
                'max': float(max_duration),
                'range_1std': [float(avg_duration - std_duration), float(avg_duration + std_duration)],
                'range_2std': [float(avg_duration - 2*std_duration), float(avg_duration + 2*std_duration)]
            }
        
        self.statistics = stats
        return stats
    
    def plot_distributions(self):
        """Plot distributions of drawdown depth and duration"""
        if not hasattr(self, 'drawdowns_df') or len(self.drawdowns_df) == 0:
            print("No drawdowns to plot.")
            return
        
        df = self.drawdowns_df.copy()
        df['max_drawdown_pct'] = df['max_drawdown'] * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Distribution of maximum drawdown depth
        axes[0, 0].hist(df['max_drawdown_pct'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df['max_drawdown_pct'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["max_drawdown_pct"].mean():.2f}%')
        axes[0, 0].axvline(df['max_drawdown_pct'].median(), color='green', linestyle='--', 
                          label=f'Median: {df["max_drawdown_pct"].median():.2f}%')
        axes[0, 0].set_xlabel('Maximum Drawdown (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Maximum Drawdown Depth')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribution of days to trough
        axes[0, 1].hist(df['days_to_trough'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(df['days_to_trough'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["days_to_trough"].mean():.1f} days')
        axes[0, 1].axvline(df['days_to_trough'].median(), color='green', linestyle='--', 
                          label=f'Median: {df["days_to_trough"].median():.1f} days')
        axes[0, 1].set_xlabel('Days to Trough')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Days to Trough')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scatter: Depth vs Days to Trough
        axes[1, 0].scatter(df['days_to_trough'], df['max_drawdown_pct'], alpha=0.6)
        axes[1, 0].set_xlabel('Days to Trough')
        axes[1, 0].set_ylabel('Maximum Drawdown (%)')
        axes[1, 0].set_title('Drawdown Depth vs Time to Trough')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Total duration (if available)
        recovered = df[df['total_duration'].notna()]
        if len(recovered) > 0:
            axes[1, 1].hist(recovered['total_duration'], bins=30, edgecolor='black', alpha=0.7, color='purple')
            axes[1, 1].axvline(recovered['total_duration'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {recovered["total_duration"].mean():.1f} days')
            axes[1, 1].axvline(recovered['total_duration'].median(), color='green', linestyle='--', 
                              label=f'Median: {recovered["total_duration"].median():.1f} days')
            axes[1, 1].set_xlabel('Total Duration (days)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Total Duration (Peak to Recovery)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No recovery data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Total Duration (No Data)')
        
        plt.tight_layout()
        plt.savefig('drawdown_depth_duration_analysis.png', dpi=150)
        print("\nPlots saved to drawdown_depth_duration_analysis.png")
        plt.show()
    
    def save_results(self, output_file='drawdown_depth_duration_stats.json'):
        """Save analysis results"""
        if hasattr(self, 'statistics'):
            with open(output_file, 'w') as f:
                json.dump(self.statistics, f, indent=2, default=str)
            print(f"\nStatistics saved to {output_file}")
        
        if hasattr(self, 'drawdowns_df'):
            self.drawdowns_df.to_csv('drawdown_details.csv', index=False)
            print(f"Drawdown details saved to drawdown_details.csv")


def main():
    """Main analysis function"""
    csv_file = 'ASI_close_prices_last_15_years_2025-12-05.csv'
    
    # Initialize analyzer
    analyzer = DrawdownDepthDurationAnalyzer(csv_file)
    
    # Identify drawdowns
    drawdowns = analyzer.identify_drawdowns(threshold=-0.05, lookahead_days=10)
    
    # Analyze statistics
    stats = analyzer.analyze_statistics()
    
    # Plot distributions
    analyzer.plot_distributions()
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

