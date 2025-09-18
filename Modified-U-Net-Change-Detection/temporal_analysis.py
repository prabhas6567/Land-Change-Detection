import numpy as np
from datetime import datetime
import pandas as pd
from scipy.stats import linregress

class TemporalAnalysis:
    def __init__(self):
        self.change_history = []
        
    def analyze_temporal_changes(self, change_masks, timestamps):
        """Analyze changes over time"""
        if len(change_masks) != len(timestamps):
            raise ValueError("Number of masks must match number of timestamps")
            
        change_rates = []
        for i in range(len(change_masks)-1):
            rate = self._calculate_change_rate(change_masks[i], 
                                            change_masks[i+1],
                                            timestamps[i],
                                            timestamps[i+1])
            change_rates.append(rate)
            
        trend = self._analyze_change_trend(change_rates)
        seasonality = self._detect_seasonality(change_rates, timestamps)
        
        return {
            'change_rates': change_rates,
            'trend': trend,
            'seasonality': seasonality,
            'forecast': self._forecast_changes(change_rates, timestamps)
        }
    
    def _calculate_change_rate(self, mask1, mask2, time1, time2):
        """Calculate rate of change between two timestamps"""
        time_delta = (datetime.strptime(time2, '%Y-%m-%d') - 
                     datetime.strptime(time1, '%Y-%m-%d')).days
        
        change_area = np.sum(np.logical_xor(mask1, mask2))
        return change_area / time_delta if time_delta > 0 else 0
    
    def _analyze_change_trend(self, rates):
        """Analyze overall trend in change rates"""
        if len(rates) < 2:
            return None
            
        x = np.arange(len(rates))
        slope, intercept, r_value, p_value, std_err = linregress(x, rates)
        
        return {
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'confidence': 1 - p_value
        }