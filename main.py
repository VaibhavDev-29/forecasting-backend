import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class CropForecastingModule:
    """Main forecasting module for the FairChain platform that provides crop forecasts
    using various time series models."""
    
    def __init__(self, data_path=None, df=None):
        """Initialize with either a path to CSV data or a pandas DataFrame"""
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Either data_path or df must be provided")
        
        self.processed_data = None
        self.forecasts = {}
        self.models = {}
        
    def preprocess_data(self):
        """Process the raw agricultural data into time series format"""
        # Extract the column names related to years
        year_columns = [col for col in self.df.columns if '-' in col]
        
        # Create an empty list to store processed data
        processed_data = []
        
        # Extract actual crop names from the dataset
        # In this dataset, crop names appear after numbers like "1. Arhar/Tur", "2. Bajra", etc.
        crop_names = []
        for _, row in self.df.iterrows():
            if pd.isna(row['State/Crop/District']) or not isinstance(row['State/Crop/District'], str):
                continue
                
            value = row['State/Crop/District'].strip()
            
            # Match pattern like "1. Rice", "2. Wheat", etc.
            if value.strip().startswith(tuple([f"{i}." for i in range(1, 50)])):
                parts = value.split('.')
                if len(parts) >= 2:
                    # Extract the crop name, removing any initial digits and dots
                    crop_name = parts[1].strip()
                    if crop_name not in crop_names and not crop_name.lower() in ['buxar', 'bhojpur', 'bhagalpur', 'bihar']:
                        crop_names.append(crop_name)
            
            # Also extract from "Total (Crop)" format
            elif value.startswith("Total (") and value.endswith(")"):
                crop_name = value.replace("Total (", "").replace(")", "").strip()
                if crop_name not in crop_names and not crop_name.lower() in ['buxar', 'bhojpur', 'bhagalpur', 'bihar']:
                    crop_names.append(crop_name)
        
        # Sort crop names for better display
        crop_names.sort()
        print(f"Found {len(crop_names)} crops in dataset: {', '.join(crop_names)}")
        
        # Now process the data rows
        current_crop = None
        for _, row in self.df.iterrows():
            # Skip rows with empty or invalid State/Crop/District
            if pd.isna(row['State/Crop/District']) or not isinstance(row['State/Crop/District'], str):
                continue
                
            value = row['State/Crop/District'].strip()
            
            # Identify crop headers
            if value.startswith(tuple([f"{i}." for i in range(1, 50)])) and "Buxar" not in value:
                parts = value.split('.')
                if len(parts) >= 2:
                    potential_crop = parts[1].strip()
                    if potential_crop in crop_names:
                        current_crop = potential_crop
                        continue
            
            # Also check for "Total (Crop)" format to identify the current crop
            elif value.startswith("Total (") and value.endswith(")"):
                potential_crop = value.replace("Total (", "").replace(")", "").strip()
                if potential_crop in crop_names:
                    current_crop = potential_crop
                    continue
            
            # Skip rows that aren't district data (like headers)
            if "Buxar" not in value and "1. Buxar" not in value:
                continue
                
            # Skip if we haven't identified a crop yet
            if current_crop is None:
                continue
                
            # Get season
            if pd.notna(row['Season']):
                season = row['Season']
            else:
                season = "Unknown"
                
            # Iterate through each year column
            for year in year_columns:
                # Get the Area, Production, and Yield for this year
                # Each year has 3 columns (Area, Production, Yield)
                year_start = int(year.split(' - ')[0])
                year_idx = self.df.columns.get_loc(year)
                
                try:
                    area = row.iloc[year_idx]
                    production = row.iloc[year_idx + 1]
                    yield_value = row.iloc[year_idx + 2]
                    
                    # Only add data points that have valid values
                    if pd.notna(area) and pd.notna(production) and pd.notna(yield_value):
                        # Convert to numeric and skip if conversion fails
                        try:
                            area = float(area)
                            production = float(production)
                            yield_value = float(yield_value)
                            
                            processed_data.append({
                                'Crop': current_crop,
                                'Season': season,
                                'Year': year_start,
                                'Area': area,
                                'Production': production,
                                'Yield': yield_value
                            })
                        except (ValueError, TypeError):
                            # Skip rows where conversion to float fails
                            continue
                except IndexError:
                    # Skip if columns are not available
                    continue
        
        # Convert the list to a DataFrame
        self.processed_data = pd.DataFrame(processed_data)
        
        if len(self.processed_data) == 0:
            print("WARNING: No data was processed. Check dataset format.")
            return self.processed_data
            
        # Print summary of processed data
        unique_crops = self.processed_data['Crop'].unique()
        print(f"Processed data contains {len(unique_crops)} crops: {', '.join(unique_crops)}")
        
        # Convert Year to datetime for time series analysis
        self.processed_data['Date'] = pd.to_datetime(self.processed_data['Year'], format='%Y')
        
        return self.processed_data
    
    def visualize_crop_trends(self, crop_name=None, metric='Production'):
        """Visualize trends for a specific crop or all crops"""
        if self.processed_data is None:
            self.preprocess_data()
            
        plt.figure(figsize=(15, 8))
        
        if crop_name:
            # Filter data for the specified crop
            crop_data = self.processed_data[self.processed_data['Crop'] == crop_name]
            
            # Group by year and calculate mean for the metric
            yearly_data = crop_data.groupby('Year')[metric].mean()
            
            plt.plot(yearly_data.index, yearly_data.values, marker='o', linewidth=2, label=crop_name)
            plt.title(f'{metric} Trend for {crop_name}')
        else:
            # Get top 5 crops by total production
            top_crops = self.processed_data.groupby('Crop')[metric].sum().nlargest(5).index
            
            for crop in top_crops:
                crop_data = self.processed_data[self.processed_data['Crop'] == crop]
                yearly_data = crop_data.groupby('Year')[metric].mean()
                plt.plot(yearly_data.index, yearly_data.values, marker='o', linewidth=2, label=crop)
            
            plt.title(f'{metric} Trends for Top 5 Crops')
        
        plt.xlabel('Year')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        return plt
    
    def decompose_time_series(self, crop_name, metric='Production'):
        """Decompose time series into trend, seasonal, and residual components"""
        if self.processed_data is None:
            self.preprocess_data()
            
        # Filter data for the specified crop
        crop_data = self.processed_data[self.processed_data['Crop'] == crop_name]
        
        # Group by year and calculate mean for the metric
        yearly_data = crop_data.groupby('Year')[metric].mean()
        
        # Ensure we have enough data points for decomposition
        if len(yearly_data) < 4:
            return None
        
        # Perform decomposition
        decomposition = seasonal_decompose(yearly_data, model='additive', period=min(4, len(yearly_data)))
        
        # Plot decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
        
        ax1.plot(decomposition.observed.index, decomposition.observed.values)
        ax1.set_title('Observed')
        
        ax2.plot(decomposition.trend.index, decomposition.trend.values)
        ax2.set_title('Trend')
        
        ax3.plot(decomposition.seasonal.index, decomposition.seasonal.values)
        ax3.set_title('Seasonal')
        
        ax4.plot(decomposition.resid.index, decomposition.resid.values)
        ax4.set_title('Residual')
        
        plt.tight_layout()
        
        return plt, decomposition
    
    def train_prophet_model(self, crop_name, metric='Production', periods=5):
        """Train and forecast using Facebook Prophet model"""
        if self.processed_data is None:
            self.preprocess_data()
            
        # Filter data for the specified crop
        crop_data = self.processed_data[self.processed_data['Crop'] == crop_name]
        
        # Ensure data is numeric
        crop_data[metric] = pd.to_numeric(crop_data[metric], errors='coerce')
        
        # Create Prophet dataframe with 'ds' and 'y' columns
        prophet_df = crop_data.groupby('Date').agg({metric: 'mean'}).reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Check if we have enough data
        if len(prophet_df) < 2:
            print(f"Insufficient data for Prophet model for {crop_name}. Need at least 2 data points.")
            return None, None, None
            
        # Train Prophet model
        try:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(prophet_df)
            
            # Make future dataframe and forecast using 'YE' frequency instead of deprecated 'Y'
            # Create custom future dates to avoid using deprecated 'Y' frequency
            last_date = prophet_df['ds'].max()
            future_dates = [last_date + pd.DateOffset(years=i) for i in range(1, periods + 1)]
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Generate forecast
            forecast = model.predict(future_df)
            
            # Store model and forecast
            model_key = f"{crop_name}_{metric}_prophet"
            self.models[model_key] = model
            self.forecasts[model_key] = forecast
            
            # Plot forecast
            plt.figure(figsize=(15, 8))
            model.plot(forecast, figsize=(15, 8))
            plt.title(f'Prophet Forecast for {crop_name} - {metric}')
            
            return model, forecast, plt
        except Exception as e:
            print(f"Error training Prophet model for {crop_name}: {str(e)}")
            return None, None, None
    
    def train_sarima_model(self, crop_name, metric='Production', periods=5):
        """Train and forecast using SARIMA model"""
        if self.processed_data is None:
            self.preprocess_data()
            
        try:
            # Filter data for the specified crop
            crop_data = self.processed_data[self.processed_data['Crop'] == crop_name]
            
            # Ensure data is numeric
            crop_data[metric] = pd.to_numeric(crop_data[metric], errors='coerce')
            
            # Group by year and calculate mean for the metric
            yearly_data = crop_data.groupby('Year')[metric].mean()
            
            # Remove any NaN values
            yearly_data = yearly_data.dropna()
            
            # Need at least 5 data points for a meaningful SARIMA model
            if len(yearly_data) < 5:
                print(f"Insufficient data for SARIMA model for {crop_name}")
                return None, None, None
            
            # Train SARIMA model
            # p, d, q parameters for the non-seasonal part
            # P, D, Q, s parameters for the seasonal part
            try:
                # Use simpler model parameters if we have limited data
                if len(yearly_data) < 8:
                    model = SARIMAX(yearly_data.values, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
                else:
                    model = SARIMAX(yearly_data.values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
                    
                results = model.fit(disp=False)
                
                # Forecast future values
                forecast = results.get_forecast(steps=periods)
                
                # Create forecast index (years) using integers instead of datetime
                last_year = yearly_data.index[-1]
                forecast_index = range(last_year + 1, last_year + periods + 1)
                forecast_values = forecast.predicted_mean
                
                # Calculate confidence intervals
                conf_int = forecast.conf_int()
                
                # Store model and forecast
                model_key = f"{crop_name}_{metric}_sarima"
                self.models[model_key] = results
                self.forecasts[model_key] = {
                    'forecast': forecast_values,
                    'index': forecast_index,
                    'conf_int': conf_int
                }
                
                # Plot forecast
                plt.figure(figsize=(15, 8))
                plt.plot(yearly_data.index, yearly_data.values, label='Historical Data')
                plt.plot(forecast_index, forecast_values, label='SARIMA Forecast', color='red')
                
                # Check if conf_int is DataFrame or ndarray and plot accordingly
                if isinstance(conf_int, pd.DataFrame):
                    lower_bound = conf_int.iloc[:, 0]
                    upper_bound = conf_int.iloc[:, 1]
                else:
                    # Handle case where conf_int is ndarray
                    lower_bound = conf_int[:, 0]
                    upper_bound = conf_int[:, 1]
                
                plt.fill_between(forecast_index, lower_bound, upper_bound, color='pink', alpha=0.3)
                
                plt.title(f'SARIMA Forecast for {crop_name} - {metric}')
                plt.xlabel('Year')
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                return results, forecast, plt
            except Exception as e:
                print(f"Error in SARIMA fitting for {crop_name}: {str(e)}")
                return None, None, None
        except Exception as e:
            print(f"Could not train SARIMA model for {crop_name}. {str(e)}")
            return None, None, None
    
    def train_xgboost_model(self, crop_name, metric='Production', periods=5):
        """Train and forecast using XGBoost model with engineered features"""
        if self.processed_data is None:
            self.preprocess_data()
            
        try:
            # Filter data for the specified crop
            crop_data = self.processed_data[self.processed_data['Crop'] == crop_name]
            
            # Ensure all numeric columns are properly formatted
            numeric_cols = [metric, 'Area', 'Yield']
            for col in numeric_cols:
                crop_data[col] = pd.to_numeric(crop_data[col], errors='coerce')
            
            # Filter out rows with NaN values after conversion
            crop_data = crop_data.dropna(subset=numeric_cols)
            
            if len(crop_data) < 5:  # Need sufficient data
                print(f"Insufficient data for XGBoost model for {crop_name}")
                return None, None, None
            
            # Feature engineering for time series
            # Group by year and calculate stats for the metric
            ts_data = crop_data.groupby('Year').agg({
                metric: ['mean', 'sum', 'count'],
                'Area': ['mean', 'sum'],
                'Yield': ['mean']
            })
            
            # Flatten multi-index columns
            ts_data.columns = ['_'.join(col).strip() for col in ts_data.columns.values]
            
            # Create lag features (t-1, t-2)
            for col in ts_data.columns:
                for lag in range(1, min(3, len(ts_data) - 1)):
                    ts_data[f'{col}_lag{lag}'] = ts_data[col].shift(lag)
            
            # Create rolling mean features
            for window in [2, 3]:
                if len(ts_data) > window:
                    for col in [f"{metric}_mean", f"{metric}_sum", "Area_mean", "Yield_mean"]:
                        ts_data[f'{col}_rolling{window}'] = ts_data[col].rolling(window=window).mean()
            
            # Drop NaN values
            ts_data = ts_data.dropna()
            
            if len(ts_data) < 5:  # Need sufficient data for training
                print(f"Insufficient data for XGBoost model for {crop_name} after feature engineering")
                return None, None, None
                
            # Prepare features and target
            X = ts_data.drop(columns=[f"{metric}_mean"])
            y = ts_data[f"{metric}_mean"]
            
            # Train XGBoost model
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
            model.fit(X, y)
            
            # Forecast iteratively for future periods
            forecast_values = []
            last_data = ts_data.iloc[-1:].copy()
            
            for i in range(periods):
                # Predict next value
                next_value = model.predict(last_data.drop(columns=[f"{metric}_mean"]))[0]
                forecast_values.append(next_value)
                
                # Create new row with forecasted value
                new_row = last_data.copy()
                new_row.index = [new_row.index[0] + 1]  # Increment year
                new_row[f"{metric}_mean"] = next_value
                
                # Update lagged values for next prediction
                for col in ts_data.columns:
                    if "_lag" in col:
                        lag_num = int(col.split("lag")[1])
                        orig_col = col.split("_lag")[0]
                        if lag_num == 1:
                            new_row[col] = last_data[orig_col].values[0]
                        else:
                            new_row[col] = last_data[f"{orig_col}_lag{lag_num-1}"].values[0]
                
                # Update rolling means
                for window in [2, 3]:
                    for col in [f"{metric}_mean", f"{metric}_sum", "Area_mean", "Yield_mean"]:
                        if f'{col}_rolling{window}' in new_row.columns:
                            values_to_average = []
                            for w in range(window):
                                if w == 0:
                                    values_to_average.append(new_row[col].values[0])
                                else:
                                    idx = last_data.index[0] - w + 1
                                    if idx in ts_data.index:
                                        values_to_average.append(ts_data.loc[idx, col])
                            new_row[f'{col}_rolling{window}'] = np.mean(values_to_average)
                
                # Add to historical data
                last_data = new_row
            
            # Create forecast index (years)
            last_year = ts_data.index[-1]
            forecast_index = range(last_year + 1, last_year + periods + 1)
            
            # Store model and forecast
            model_key = f"{crop_name}_{metric}_xgboost"
            self.models[model_key] = model
            self.forecasts[model_key] = {
                'forecast': forecast_values,
                'index': forecast_index
            }
            
            # Plot forecast
            plt.figure(figsize=(15, 8))
            plt.plot(ts_data.index, ts_data[f"{metric}_mean"], label='Historical Data')
            plt.plot(forecast_index, forecast_values, label='XGBoost Forecast', color='green')
            plt.title(f'XGBoost Forecast for {crop_name} - {metric}')
            plt.xlabel('Year')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            return model, forecast_values, plt
        except Exception as e:
            print(f"Error in XGBoost model for {crop_name}: {str(e)}")
            return None, None, None
    
    def train_lstm_model(self, crop_name, metric='Production', periods=5, sequence_length=3):
        """Train and forecast using LSTM neural network"""
        if self.processed_data is None:
            self.preprocess_data()
            
        try:
            # Suppress warnings temporarily
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Filter data for the specified crop
                crop_data = self.processed_data[self.processed_data['Crop'] == crop_name]
                
                # Ensure data is numeric
                crop_data[metric] = pd.to_numeric(crop_data[metric], errors='coerce')
                
                # Group by year and calculate mean for the metric
                yearly_data = crop_data.groupby('Year')[metric].mean().reset_index()
                
                # Remove any NaN values
                yearly_data = yearly_data.dropna()
                
                if len(yearly_data) < sequence_length + 5:  # Need sufficient data for sequences
                    print(f"Insufficient data for LSTM model for {crop_name}")
                    return None, None, None
                
                # Scale data
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(yearly_data[[metric]])
                
                # Create sequences
                X, y = [], []
                for i in range(len(scaled_data) - sequence_length):
                    X.append(scaled_data[i:i+sequence_length])
                    y.append(scaled_data[i+sequence_length])
                
                X, y = np.array(X), np.array(y)
                
                # Reshape for LSTM [samples, time steps, features]
                X = X.reshape(X.shape[0], X.shape[1], 1)
                
                # Silence Keras warnings
                import tensorflow as tf
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                
                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                    Dropout(0.2),
                    LSTM(50),
                    Dropout(0.2),
                    Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X, y, epochs=100, batch_size=8, verbose=0)
                
                # Forecast future values
                forecast_values = []
                current_batch = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
                
                for i in range(periods):
                    # Predict next value
                    next_value = model.predict(current_batch, verbose=0)[0]
                    forecast_values.append(next_value[0])
                    
                    # Update batch for next prediction
                    current_batch = np.append(current_batch[:, 1:, :], [[next_value]], axis=1)
                
                # Inverse transform the forecasted values
                forecast_values = scaler.inverse_transform(np.array(forecast_values).reshape(-1, 1))
                
                # Create forecast index (years)
                last_year = yearly_data['Year'].iloc[-1]
                forecast_index = range(last_year + 1, last_year + periods + 1)
                
                # Store model and forecast
                model_key = f"{crop_name}_{metric}_lstm"
                self.models[model_key] = model
                self.forecasts[model_key] = {
                    'forecast': forecast_values,
                    'index': forecast_index
                }
                
                # Plot forecast
                plt.figure(figsize=(15, 8))
                plt.plot(yearly_data['Year'], yearly_data[metric], label='Historical Data')
                plt.plot(forecast_index, forecast_values, label='LSTM Forecast', color='purple')
                plt.title(f'LSTM Forecast for {crop_name} - {metric}')
                plt.xlabel('Year')
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                return model, forecast_values, plt
        except Exception as e:
            print(f"Error in LSTM model for {crop_name}: {str(e)}")
            return None, None, None
    
    def ensemble_forecast(self, crop_name, metric='Production', periods=5, models_to_use=['prophet', 'sarima', 'xgboost', 'lstm']):
        """Create an ensemble forecast combining multiple models"""
        forecasts = []
        model_names = []
        
        # Train each model if not already trained
        if 'prophet' in models_to_use:
            model_key = f"{crop_name}_{metric}_prophet"
            if model_key not in self.forecasts:
                model, _, _ = self.train_prophet_model(crop_name, metric, periods)
                if model is None:
                    # If model training failed, remove from models_to_use
                    models_to_use.remove('prophet')
            if model_key in self.forecasts:
                try:
                    # Extract yhat from Prophet forecast
                    prophet_forecast = self.forecasts[model_key]
                    future_forecast = prophet_forecast[prophet_forecast['ds'] > prophet_forecast['ds'].max() - pd.DateOffset(years=periods)]
                    forecasts.append(future_forecast['yhat'].values)
                    model_names.append('prophet')
                except Exception as e:
                    print(f"Error extracting Prophet forecast: {str(e)}")
        
        if 'sarima' in models_to_use:
            model_key = f"{crop_name}_{metric}_sarima"
            if model_key not in self.forecasts:
                model, _, _ = self.train_sarima_model(crop_name, metric, periods)
                if model is None:
                    # If model training failed, remove from models_to_use
                    models_to_use.remove('sarima')
            if model_key in self.forecasts:
                try:
                    forecasts.append(self.forecasts[model_key]['forecast'])
                    model_names.append('sarima')
                except Exception as e:
                    print(f"Error extracting SARIMA forecast: {str(e)}")
        
        if 'xgboost' in models_to_use:
            model_key = f"{crop_name}_{metric}_xgboost"
            if model_key not in self.forecasts:
                model, _, _ = self.train_xgboost_model(crop_name, metric, periods)
                if model is None:
                    # If model training failed, remove from models_to_use
                    models_to_use.remove('xgboost')
            if model_key in self.forecasts:
                try:
                    forecasts.append(self.forecasts[model_key]['forecast'])
                    model_names.append('xgboost')
                except Exception as e:
                    print(f"Error extracting XGBoost forecast: {str(e)}")
        
        if 'lstm' in models_to_use:
            model_key = f"{crop_name}_{metric}_lstm"
            if model_key not in self.forecasts:
                model, _, _ = self.train_lstm_model(crop_name, metric, periods)
                if model is None:
                    # If model training failed, remove from models_to_use
                    models_to_use.remove('lstm')
            if model_key in self.forecasts:
                try:
                    # Check if forecast is 2D and needs flattening
                    forecast_values = self.forecasts[model_key]['forecast']
                    if isinstance(forecast_values, np.ndarray) and forecast_values.ndim > 1:
                        forecasts.append(forecast_values.flatten())
                    else:
                        forecasts.append(forecast_values)
                    model_names.append('lstm')
                except Exception as e:
                    print(f"Error extracting LSTM forecast: {str(e)}")
        
        # If no forecasts were generated, return None
        if not forecasts:
            print(f"No valid forecasts could be generated for {crop_name}")
            return None, None, None
        
        try:
            # Create ensemble by averaging forecasts
            # First, ensure all forecasts have the same length
            min_length = min(len(f) for f in forecasts)
            forecasts = [f[:min_length] for f in forecasts]
            
            # Calculate ensemble forecast (average)
            ensemble_forecast = np.mean(forecasts, axis=0)
            
            # Create forecast index
            crop_data = self.processed_data[self.processed_data['Crop'] == crop_name]
            last_year = crop_data['Year'].max()
            forecast_index = range(last_year + 1, last_year + min_length + 1)
            
            # Store ensemble forecast
            model_key = f"{crop_name}_{metric}_ensemble"
            self.forecasts[model_key] = {
                'forecast': ensemble_forecast,
                'index': forecast_index,
                'individual_forecasts': forecasts,
                'model_names': model_names
            }
            
            # Plot ensemble forecast with individual model forecasts
            plt.figure(figsize=(15, 8))
            
            # Plot historical data
            yearly_data = crop_data.groupby('Year')[metric].mean()
            plt.plot(yearly_data.index, yearly_data.values, label='Historical Data', color='blue')
            
            # Plot individual model forecasts
            colors = ['red', 'green', 'purple', 'orange']
            for i, model_name in enumerate(model_names):
                if i < len(forecasts):
                    plt.plot(forecast_index, forecasts[i], label=f'{model_name.capitalize()} Forecast', 
                             color=colors[i % len(colors)], linestyle='--', alpha=0.7)
            
            # Plot ensemble forecast
            plt.plot(forecast_index, ensemble_forecast, label='Ensemble Forecast', 
                     color='black', linewidth=2)
            
            plt.title(f'Ensemble Forecast for {crop_name} - {metric}')
            plt.xlabel('Year')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            return ensemble_forecast, forecast_index, plt
        except Exception as e:
            print(f"Error creating ensemble forecast for {crop_name}: {str(e)}")
            return None, None, None
    
    def get_top_crops(self, metric='Production', top_n=5):
        """Get top N crops by specified metric"""
        if self.processed_data is None:
            self.preprocess_data()
        
        # Ensure the metric column is numeric
        self.processed_data[metric] = pd.to_numeric(self.processed_data[metric], errors='coerce')
        
        # Group by crop and sum the metric, then get top N
        top_crops = self.processed_data.groupby('Crop')[metric].sum().nlargest(top_n)
        return top_crops
    
    def forecast_regional_demand(self, region='Buxar', top_n=5, periods=5):
        """Forecast demand for top N crops in a specific region"""
        if self.processed_data is None:
            self.preprocess_data()
            
        # List of districts/regions to exclude from crop names
        exclude_terms = ['buxar', 'bhojpur', 'bhagalpur', 'bihar', 'district', 'total']
            
        # Ensure we're working with the specified region
        region_data = self.processed_data[self.processed_data['Crop'].str.contains(region, case=False, na=False)]
        
        # If we don't have region-specific data, use all data
        if len(region_data) == 0:
            region_data = self.processed_data
            
        # Filter out any rows where Crop contains district/region names
        filtered_data = region_data[~region_data['Crop'].str.lower().str.contains('|'.join(exclude_terms), case=False, na=False)]
        
        # If no crops left after filtering, fall back to original data
        if len(filtered_data) == 0:
            filtered_data = region_data
            
        # Get top crops by production
        top_crops = filtered_data.groupby('Crop')['Production'].sum().nlargest(top_n).index.tolist()
        
        results = {}
        for crop in top_crops:
            # Skip if crop name contains a district/region name
            if any(term in crop.lower() for term in exclude_terms):
                continue
                
            try:
                # Generate ensemble forecast for this crop
                ensemble_forecast, forecast_index, _ = self.ensemble_forecast(crop, 'Production', periods)
                
                # Only add to results if forecast was successfully generated
                if ensemble_forecast is not None and forecast_index is not None:
                    results[crop] = {
                        'forecast': ensemble_forecast,
                        'years': forecast_index
                    }
            except Exception as e:
                print(f"Failed to generate forecast for {crop}: {str(e)}")
                continue
                
        # If we couldn't generate forecasts for any crop, try with a simpler approach
        if not results:
            for crop in top_crops:
                # Skip if crop name contains a district/region name
                if any(term in crop.lower() for term in exclude_terms):
                    continue
                
                try:
                    # Get historical data for this crop
                    crop_data = filtered_data[filtered_data['Crop'] == crop]
                    
                    # Calculate average yearly increase
                    if len(crop_data) > 1:
                        yearly_data = crop_data.groupby('Year')['Production'].mean()
                        years = sorted(yearly_data.index)
                        
                        if len(years) >= 2:
                            # Simple trending forecast
                            last_value = yearly_data[years[-1]]
                            avg_increase = (yearly_data[years[-1]] - yearly_data[years[0]]) / (years[-1] - years[0])
                            
                            # Generate forecast
                            forecast_values = []
                            forecast_years = []
                            
                            for i in range(1, periods + 1):
                                forecast_year = years[-1] + i
                                forecast_value = last_value + (avg_increase * i)
                                # Ensure no negative values
                                forecast_value = max(0, forecast_value)
                                
                                forecast_values.append(forecast_value)
                                forecast_years.append(forecast_year)
                                
                            results[crop] = {
                                'forecast': forecast_values,
                                'years': forecast_years
                            }
                except Exception as e:
                    print(f"Failed to generate simple forecast for {crop}: {str(e)}")
                    continue
        
        return results
    
    def suggest_optimal_crops(self, region='Buxar', metric='Yield', top_n=5):
        """Suggest optimal crops to grow based on yield trends"""
        if self.processed_data is None:
            self.preprocess_data()
        
        try:
            # List of districts/regions to exclude from crop names
            exclude_terms = ['buxar', 'bhojpur', 'bhagalpur', 'bihar', 'district', 'total']
            
            # MODIFIED: Instead of filtering by region, return all crops if region not found
            if region and region.lower() != 'all':
                region_data = self.processed_data[self.processed_data['Crop'].str.contains(region, case=False, na=False)]
                if len(region_data) > 0:
                    filtered_data = region_data
                else:
                    # Use all data and print that region isn't found
                    print(f"Region '{region}' not found in data. Using all available crop data.")
                    filtered_data = self.processed_data
            else:
                filtered_data = self.processed_data
            
            # Filter out any rows where Crop contains district/region names
            valid_crops = filtered_data[~filtered_data['Crop'].str.lower().str.contains('|'.join(exclude_terms), case=False, na=False)]
            
            # Print unique crops for debugging
            unique_crops = valid_crops['Crop'].unique()
            print(f"Found {len(unique_crops)} valid crops for analysis: {', '.join(unique_crops)}")
            
            # Get recent data (last 5 years)
            recent_years = valid_crops['Year'].unique()
            if len(recent_years) > 5:
                recent_years = sorted(recent_years)[-5:]
            
            # Make a copy of the data to avoid modifying the original
            data_copy = valid_crops.copy()
            
            # Convert metric column to numeric
            data_copy[metric] = pd.to_numeric(data_copy[metric], errors='coerce')
            
            # Filter recent data and drop NaN values
            recent_data = data_copy[data_copy['Year'].isin(recent_years)]
            recent_data = recent_data.dropna(subset=[metric])
            
            if len(recent_data) == 0:
                print(f"No valid data available for {metric} in recent years")
                return pd.Series(dtype='float64'), {}
            
            # Calculate average yield for each crop in recent years
            crop_yields = recent_data.groupby('Crop')[metric].mean().sort_values(ascending=False)
            
            # Get top N crops by yield
            top_crops = crop_yields.head(top_n)
            
            if top_crops.empty:
                print(f"No data available for {metric} to suggest optimal crops")
                return pd.Series(dtype='float64'), {}
            
            # Make forecasts for these crops
            forecasts = {}
            for crop in top_crops.index:
                try:
                    model, _, plt = self.train_prophet_model(crop, metric, periods=5)
                    if model is not None:
                        forecasts[crop] = {
                            'current_yield': top_crops[crop],
                            'forecast': self.forecasts[f"{crop}_{metric}_prophet"]
                        }
                except Exception as e:
                    print(f"Could not create forecast for {crop}: {str(e)}")
                    continue
            
            return top_crops, forecasts
        except Exception as e:
            print(f"Error suggesting optimal crops: {str(e)}")
            return pd.Series(dtype='float64'), {}
    
    def generate_crop_calendar(self, region='Buxar'):
        """Generate planting and harvesting calendar based on historical data"""
        if self.processed_data is None:
            self.preprocess_data()
        
        # Get seasonal information for crops
        crop_seasons = self.processed_data[['Crop', 'Season']].drop_duplicates()
        
        # Define typical planting and harvesting months for each season in North India
        season_calendar = {
            'Kharif': {'planting': 'June-July', 'harvesting': 'October-November'},
            'Rabi': {'planting': 'October-November', 'harvesting': 'March-April'},
            'Summer': {'planting': 'March-April', 'harvesting': 'June-July'},
            'Autumn': {'planting': 'August-September', 'harvesting': 'November-December'},
            'Winter': {'planting': 'October-November', 'harvesting': 'February-March'},
            'Whole Year': {'planting': 'Multiple seasons', 'harvesting': 'Multiple seasons'}
        }
        
        # Create calendar for each crop
        crop_calendar = {}
        for _, row in crop_seasons.iterrows():
            crop = row['Crop']
            season = row['Season']
            
            if season in season_calendar:
                if crop not in crop_calendar:
                    crop_calendar[crop] = []
                
                crop_calendar[crop].append({
                    'season': season,
                    'planting_time': season_calendar[season]['planting'],
                    'harvesting_time': season_calendar[season]['harvesting']
                })
        
        return crop_calendar
    
    def forecasting_api_response(self, crop_name, metric='Production'):
        """Generate a standardized API response for forecasts"""
        if self.processed_data is None:
            self.preprocess_data()
            
        if crop_name not in self.processed_data['Crop'].unique():
            return {
                'status': 'error',
                'message': f'Crop {crop_name} not found in dataset'
            }
        
        try:
            # Get historical data
            crop_data = self.processed_data[self.processed_data['Crop'] == crop_name]
            
            # Ensure numeric data
            numeric_cols = [metric, 'Area', 'Yield']
            for col in numeric_cols:
                crop_data[col] = pd.to_numeric(crop_data[col], errors='coerce')
                
            # Drop rows with NaN values after conversion
            crop_data = crop_data.dropna(subset=numeric_cols)
            
            # Group by year
            historical = crop_data.groupby('Year').agg({
                metric: 'mean',
                'Area': 'mean',
                'Yield': 'mean'
            }).reset_index()
            
            # Generate ensemble forecast
            ensemble_forecast, forecast_years, _ = self.ensemble_forecast(crop_name, metric)
            
            # Format response
            response = {
                'status': 'success',
                'crop': crop_name,
                'metric': metric,
                'historical_data': [
                    {
                        'year': int(row['Year']),
                        metric: float(row[metric]),
                        'area': float(row['Area']),
                        'yield': float(row['Yield'])
                    } for _, row in historical.iterrows()
                ]
            }
            
            # Add forecast data if available
            if ensemble_forecast is not None and forecast_years is not None:
                response['forecast'] = [
                    {
                        'year': int(year),
                        metric.lower(): float(value)
                    } for year, value in zip(forecast_years, ensemble_forecast)
                ]
                response['confidence_score'] = 0.85  # Placeholder, should be calculated based on model performance
            else:
                response['forecast'] = []
                response['forecast_message'] = 'Could not generate reliable forecast'
                
            # Add crop calendar
            try:
                calendar = self.generate_crop_calendar().get(crop_name, [])
                response['crop_calendar'] = calendar
            except Exception as e:
                response['crop_calendar'] = []
                response['calendar_error'] = str(e)
                
            response['last_updated'] = datetime.now().strftime('%Y-%m-%d')
            
            return response
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }


# Demo usage of the forecasting module
def run_forecasting_demo(csv_path=None, df=None, show_plots=False):
    """Run a demonstration of the forecasting capabilities"""
    import os
    import warnings
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
    
    # Initialize forecasting module
    forecaster = CropForecastingModule(data_path=csv_path, df=df)
    
    # Preprocess data
    processed_data = forecaster.preprocess_data()
    print(f"Processed {len(processed_data)} data points")
    
    # Print unique crops in the dataset
    unique_crops = processed_data['Crop'].unique()
    print(f"\nDetected {len(unique_crops)} unique crops in the dataset:")
    print(", ".join(sorted(unique_crops)))
    
    # Get top crops
    top_crops = forecaster.get_top_crops(top_n=5)
    print("\nTop 5 crops by total production:")
    for crop, production in top_crops.items():
        print(f"- {crop}: {production:,.0f} tonnes")
    
    # Get the first available crop from top_crops
    available_crops = list(top_crops.index)
    if available_crops:
        selected_crop = available_crops[0]
        # Generate forecasts for the selected crop
        print(f"\nGenerating forecasts for {selected_crop}...")
        model, forecast, plt_obj = forecaster.train_prophet_model(selected_crop, 'Production')
        
        if show_plots and plt_obj is not None:
            plt_obj.show()
        
        # Generate ensemble forecast
        print(f"\nGenerating ensemble forecast for {selected_crop}...")
        ensemble_forecast, forecast_years, plt_obj = forecaster.ensemble_forecast(selected_crop, 'Production')
        
        if show_plots and plt_obj is not None:
            plt_obj.show()
        
        if ensemble_forecast is not None and forecast_years is not None:
            print(f"\n{selected_crop} Production Forecast:")
            for year, value in zip(forecast_years, ensemble_forecast):
                print(f"- {year}: {value:,.0f} tonnes")
    else:
        print("\nNo crops available for forecasting.")
    
    # Get crop suggestions
    print("\nSuggested crops based on yield performance:")
    top_yields, forecasts = forecaster.suggest_optimal_crops()
    if isinstance(top_yields, pd.Series) and not top_yields.empty:
        for crop, yield_value in top_yields.items():
            print(f"- {crop}: {yield_value:.2f} tonnes/hectare")
            
            # Show yield forecast plots if available
            if show_plots and crop in forecasts:
                plt.figure(figsize=(10, 6))
                plt.title(f"Yield Forecast for {crop}")
                plt.show()
    else:
        print("No yield data available for suggestions.")
    
    # Generate API response example
    if available_crops:
        print(f"\nAPI response example for {selected_crop}:")
        api_response = forecaster.forecasting_api_response(selected_crop)
        # Optionally print out the first few entries of the API response
        if show_plots:
            print(f"Status: {api_response['status']}")
            if 'forecast' in api_response and api_response['forecast']:
                print(f"Forecast years: {len(api_response['forecast'])}")
    
    return forecaster


# If running as a script
if __name__ == "__main__":
    # Suppress TensorFlow and statsmodels warnings
    import os
    import warnings
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Sample example with the provided dataset
    try:
        import plotly
        has_plotly = True
    except ImportError:
        print("Importing plotly failed. Interactive plots will not work.")
        has_plotly = False
    
    # Run the forecasting demo
    forecaster = run_forecasting_demo('dataset.csv', show_plots=False)