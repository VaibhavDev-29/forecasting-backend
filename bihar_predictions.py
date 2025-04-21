import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BiharForecastingModule:
    """Forecasting module for Bihar agricultural data that provides crop forecasts
    using various time series models."""
    
    def __init__(self, data_path="bihar-dataset.csv"):
        """Initialize with the path to the Bihar CSV data"""
        self.df = pd.read_csv(data_path)
        self.processed_data = None
        self.forecasts = {}
        self.models = {}
        
    def preprocess_data(self):
        """Process the raw Bihar agricultural data into time series format"""
        # Extract the column names related to years
        year_columns = [col for col in self.df.columns if '-' in col]
        
        # Create an empty list to store processed data
        processed_data = []
        
        # Extract actual crop names from the dataset
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
                    # Exclude district/region names
                    if crop_name not in crop_names and not crop_name.lower() in ['buxar', 'bhojpur', 'bhagalpur', 'bihar', 'araria', 'arwal', 'aurangabad', 'banka', 'begusarai',
                                                                               'darbhanga', 'gaya', 'gopalganj', 'jamui', 'jehanabad', 'kaimur', 'katihar', 'khagaria',
                                                                               'kishanganj', 'lakhisarai', 'madhepura', 'madhubani', 'munger', 'muzaffarpur', 'nalanda',
                                                                               'nawada', 'pashchim champaran', 'patna', 'purbi champaran', 'purnia', 'rohtas', 'saharsa',
                                                                               'samastipur', 'saran', 'sheikhpura', 'sheohar', 'sitamarhi', 'siwan', 'supaul', 'vaishali']:
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
        current_district = None
        
        for _, row in self.df.iterrows():
            # Skip rows with empty or invalid State/Crop/District
            if pd.isna(row['State/Crop/District']) or not isinstance(row['State/Crop/District'], str):
                continue
                
            value = row['State/Crop/District'].strip()
            
            # Identify crop headers
            if value.strip().startswith(tuple([f"{i}." for i in range(1, 50)])) and "Buxar" not in value and "District" not in value and "Bihar" != value:
                parts = value.split('.')
                if len(parts) >= 2:
                    potential_crop = parts[1].strip()
                    if potential_crop in crop_names:
                        current_crop = potential_crop
                        current_district = None
                        continue
            
            # Also check for "Total (Crop)" format to identify the current crop
            elif value.startswith("Total (") and value.endswith(")"):
                potential_crop = value.replace("Total (", "").replace(")", "").strip()
                if potential_crop in crop_names:
                    current_crop = potential_crop
                    current_district = None
                    continue
            
            # Identify district rows
            district_match = False
            district_names = ['1. Buxar', '2. Bhojpur', '3. Bhagalpur', '8. Buxar']
            for district_pattern in district_names:
                if district_pattern in value:
                    current_district = "Buxar" if "Buxar" in value else "Bhojpur" if "Bhojpur" in value else "Bhagalpur" if "Bhagalpur" in value else None
                    district_match = True
                    break
            
            if district_match and current_crop:
                # Get season
                if pd.notna(row['Season']):
                    season = row['Season']
                else:
                    season = "Unknown"
                    
                # Iterate through each year column
                for year in year_columns:
                    # Get the Area, Production, and Yield for this year
                    # Each year has 3 columns (Area, Production, Yield)
                    try:
                        year_start = int(year.split(' - ')[0])
                        year_idx = self.df.columns.get_loc(year)
                        
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
                                    'District': current_district,
                                    'Season': season,
                                    'Year': year_start,
                                    'Area': area,
                                    'Production': production,
                                    'Yield': yield_value
                                })
                            except (ValueError, TypeError):
                                # Skip rows where conversion to float fails
                                continue
                    except (IndexError, ValueError):
                        # Skip if columns are not available or year format is invalid
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
    
    def train_prophet_model(self, crop_name, district=None, metric='Production', periods=5):
        """Train and forecast using Facebook Prophet model"""
        if self.processed_data is None:
            self.preprocess_data()
            
        # Filter data for the specified crop and district if provided
        if district:
            crop_data = self.processed_data[(self.processed_data['Crop'] == crop_name) & 
                                           (self.processed_data['District'] == district)]
        else:
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
            
            # Create future dates for forecasting
            last_date = prophet_df['ds'].max()
            future_dates = [last_date + pd.DateOffset(years=i) for i in range(1, periods + 1)]
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Generate forecast
            forecast = model.predict(future_df)
            
            # Store model and forecast
            model_key = f"{crop_name}_{metric}_prophet"
            if district:
                model_key = f"{district}_{model_key}"
                
            self.models[model_key] = model
            self.forecasts[model_key] = forecast
            
            # Create plot
            fig = plt.figure(figsize=(12, 6))
            model.plot(forecast, figsize=(12, 6))
            plt.title(f'Prophet Forecast for {crop_name} - {metric}' + (f' in {district}' if district else ''))
            
            return model, forecast, fig
        except Exception as e:
            print(f"Error training Prophet model for {crop_name}: {str(e)}")
            return None, None, None
    
    def train_sarima_model(self, crop_name, district=None, metric='Production', periods=5):
        """Train and forecast using SARIMA model"""
        if self.processed_data is None:
            self.preprocess_data()
            
        try:
            # Filter data for the specified crop and district if provided
            if district:
                crop_data = self.processed_data[(self.processed_data['Crop'] == crop_name) & 
                                               (self.processed_data['District'] == district)]
            else:
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
                if district:
                    model_key = f"{district}_{model_key}"
                    
                self.models[model_key] = results
                self.forecasts[model_key] = {
                    'forecast': forecast_values,
                    'index': forecast_index,
                    'conf_int': conf_int
                }
                
                # Plot forecast
                fig = plt.figure(figsize=(12, 6))
                plt.plot(yearly_data.index, yearly_data.values, label='Historical Data')
                plt.plot(forecast_index, forecast_values, label='SARIMA Forecast', color='red')
                
                # Add confidence intervals to plot
                if isinstance(conf_int, pd.DataFrame):
                    lower_bound = conf_int.iloc[:, 0]
                    upper_bound = conf_int.iloc[:, 1]
                else:
                    # Handle case where conf_int is ndarray
                    lower_bound = conf_int[:, 0]
                    upper_bound = conf_int[:, 1]
                
                plt.fill_between(forecast_index, lower_bound, upper_bound, color='pink', alpha=0.3)
                
                plt.title(f'SARIMA Forecast for {crop_name} - {metric}' + (f' in {district}' if district else ''))
                plt.xlabel('Year')
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                return results, forecast, fig
            except Exception as e:
                print(f"Error in SARIMA fitting for {crop_name}: {str(e)}")
                return None, None, None
        except Exception as e:
            print(f"Could not train SARIMA model for {crop_name}. {str(e)}")
            return None, None, None
    
    def ensemble_forecast(self, crop_name, district=None, metric='Production', periods=5, models_to_use=['prophet', 'sarima']):
        """Create an ensemble forecast combining multiple models"""
        forecasts = []
        model_names = []
        
        # Train each model if not already trained
        if 'prophet' in models_to_use:
            model_key = f"{crop_name}_{metric}_prophet"
            if district:
                model_key = f"{district}_{model_key}"
                
            if model_key not in self.forecasts:
                model, _, _ = self.train_prophet_model(crop_name, district, metric, periods)
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
            if district:
                model_key = f"{district}_{model_key}"
                
            if model_key not in self.forecasts:
                model, _, _ = self.train_sarima_model(crop_name, district, metric, periods)
                if model is None:
                    # If model training failed, remove from models_to_use
                    models_to_use.remove('sarima')
            
            if model_key in self.forecasts:
                try:
                    forecasts.append(self.forecasts[model_key]['forecast'])
                    model_names.append('sarima')
                except Exception as e:
                    print(f"Error extracting SARIMA forecast: {str(e)}")
        
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
            if district:
                crop_data = self.processed_data[(self.processed_data['Crop'] == crop_name) & 
                                               (self.processed_data['District'] == district)]
            else:
                crop_data = self.processed_data[self.processed_data['Crop'] == crop_name]
                
            last_year = crop_data['Year'].max()
            forecast_index = range(last_year + 1, last_year + min_length + 1)
            
            # Store ensemble forecast
            model_key = f"{crop_name}_{metric}_ensemble"
            if district:
                model_key = f"{district}_{model_key}"
                
            self.forecasts[model_key] = {
                'forecast': ensemble_forecast,
                'index': forecast_index,
                'individual_forecasts': forecasts,
                'model_names': model_names
            }
            
            # Plot ensemble forecast with individual model forecasts
            fig = plt.figure(figsize=(12, 6))
            
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
            
            plt.title(f'Ensemble Forecast for {crop_name} - {metric}' + (f' in {district}' if district else ''))
            plt.xlabel('Year')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            return ensemble_forecast, forecast_index, fig
        except Exception as e:
            print(f"Error creating ensemble forecast for {crop_name}: {str(e)}")
            return None, None, None
    
    def get_top_crops(self, metric='Production', district=None, top_n=5):
        """Get top N crops by specified metric"""
        if self.processed_data is None:
            self.preprocess_data()
        
        # Ensure the metric column is numeric
        self.processed_data[metric] = pd.to_numeric(self.processed_data[metric], errors='coerce')
        
        # Filter by district if specified
        if district:
            filtered_data = self.processed_data[self.processed_data['District'] == district]
        else:
            filtered_data = self.processed_data
        
        # Get recent years (last 5 years in the dataset)
        recent_years = sorted(filtered_data['Year'].unique())[-5:]
        recent_data = filtered_data[filtered_data['Year'].isin(recent_years)]
        
        # Group by crop and sum the metric, then get top N
        top_crops = recent_data.groupby('Crop')[metric].sum().nlargest(top_n)
        return top_crops
    
    def suggest_optimal_crops(self, district='Buxar', metric='Yield', top_n=5):
        """Suggest optimal crops to grow based on yield trends"""
        if self.processed_data is None:
            self.preprocess_data()
        
        try:
            # Filter by district if specified
            if district:
                filtered_data = self.processed_data[self.processed_data['District'] == district]
                if len(filtered_data) == 0:
                    # If the specified district is not found, use all data
                    print(f"District '{district}' not found in data. Using all available crop data.")
                    filtered_data = self.processed_data
            else:
                filtered_data = self.processed_data
            
            # Get recent years (last 5 years in the dataset)
            recent_years = sorted(filtered_data['Year'].unique())[-5:]
            recent_data = filtered_data[filtered_data['Year'].isin(recent_years)]
            
            # Ensure metric column is numeric
            recent_data[metric] = pd.to_numeric(recent_data[metric], errors='coerce')
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
                    model, _, plt = self.train_prophet_model(crop, district, metric, periods=5)
                    if model is not None:
                        model_key = f"{crop}_{metric}_prophet"
                        if district:
                            model_key = f"{district}_{model_key}"
                            
                        forecasts[crop] = {
                            'current_yield': top_crops[crop],
                            'forecast': self.forecasts[model_key]
                        }
                except Exception as e:
                    print(f"Could not create forecast for {crop}: {str(e)}")
                    continue
            
            return top_crops, forecasts
        except Exception as e:
            print(f"Error suggesting optimal crops: {str(e)}")
            return pd.Series(dtype='float64'), {}
    
    def forecasting_api_response(self, crop_name, district=None, metric='Production'):
        """Generate a standardized API response for forecasts"""
        if self.processed_data is None:
            self.preprocess_data()
            
        # Check if crop exists in dataset
        if crop_name not in self.processed_data['Crop'].unique():
            return {
                'status': 'error',
                'message': f'Crop {crop_name} not found in dataset'
            }
        
        # Check if district exists (if specified)
        if district and district not in self.processed_data['District'].unique():
            return {
                'status': 'error',
                'message': f'District {district} not found in dataset'
            }
        
        try:
            # Get historical data
            if district:
                crop_data = self.processed_data[(self.processed_data['Crop'] == crop_name) & 
                                               (self.processed_data['District'] == district)]
            else:
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
            ensemble_forecast, forecast_years, _ = self.ensemble_forecast(crop_name, district, metric)
            
            # Format response
            response = {
                'status': 'success',
                'crop': crop_name,
                'district': district if district else 'All Districts',
                'metric': metric,
                'historical_data': [
                    {
                        'year': int(row['Year']),
                        metric.lower(): float(row[metric]),
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
            
    def generate_crop_calendar(self, district=None):
        """Generate planting and harvesting calendar based on historical data for Bihar crops"""
        if self.processed_data is None:
            self.preprocess_data()
        
        # Filter data by district if specified
        if district:
            district_data = self.processed_data[self.processed_data['District'] == district]
            if len(district_data) == 0:
                print(f"District '{district}' not found in data. Using all available crop data.")
                district_data = self.processed_data
        else:
            district_data = self.processed_data
            
        # Get seasonal information for crops
        crop_seasons = district_data[['Crop', 'Season']].drop_duplicates()
        
        # Define typical planting and harvesting months for each season in Bihar
        season_calendar = {
            'Kharif': {'planting': 'June-July', 'harvesting': 'October-November'},
            'Rabi': {'planting': 'October-November', 'harvesting': 'March-April'},
            'Summer': {'planting': 'March-April', 'harvesting': 'June-July'},
            'Autumn': {'planting': 'August-September', 'harvesting': 'November-December'},
            'Winter': {'planting': 'October-November', 'harvesting': 'February-March'},
            'Whole Year': {'planting': 'Multiple seasons', 'harvesting': 'Multiple seasons'},
            'Zaid': {'planting': 'February-March', 'harvesting': 'May-June'}
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

def run_bihar_forecasting_demo(output_dir="outputs"):
    """Run a demonstration of the Bihar forecasting capabilities"""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize forecasting module
    forecaster = BiharForecastingModule()
    
    # Preprocess data
    processed_data = forecaster.preprocess_data()
    print(f"Processed {len(processed_data)} data points")
    
    # Print unique crops in the dataset
    unique_crops = processed_data['Crop'].unique()
    print(f"\nDetected {len(unique_crops)} unique crops in the dataset:")
    print(", ".join(sorted(unique_crops)))
    
    # Get top crops
    top_crops = forecaster.get_top_crops(district='Buxar', top_n=5)
    print("\nTop 5 crops by total production in Buxar:")
    for crop, production in top_crops.items():
        print(f"- {crop}: {production:,.0f} tonnes")
    
    # Generate forecasts for top 3 crops
    for i, crop_name in enumerate(top_crops.index[:3]):
        print(f"\nGenerating forecasts for {crop_name} in Buxar...")
        
        # Prophet model
        _, forecast, plt_obj = forecaster.train_prophet_model(crop_name, district='Buxar', metric='Production')
        if plt_obj is not None:
            plt_obj.savefig(os.path.join(output_dir, f"{crop_name}_prophet_forecast.png"))
            plt.close(plt_obj)
        
        # SARIMA model
        _, forecast, plt_obj = forecaster.train_sarima_model(crop_name, district='Buxar', metric='Production')
        if plt_obj is not None:
            plt_obj.savefig(os.path.join(output_dir, f"{crop_name}_sarima_forecast.png"))
            plt.close(plt_obj)
        
        # Ensemble forecast
        ensemble_forecast, forecast_years, plt_obj = forecaster.ensemble_forecast(crop_name, district='Buxar', metric='Production')
        if plt_obj is not None:
            plt_obj.savefig(os.path.join(output_dir, f"{crop_name}_ensemble_forecast.png"))
            plt.close(plt_obj)
        
        if ensemble_forecast is not None and forecast_years is not None:
            print(f"\n{crop_name} Production Forecast for Buxar:")
            for year, value in zip(forecast_years, ensemble_forecast):
                print(f"- {year}: {value:,.0f} tonnes")
    
    # Get crop suggestions
    print("\nSuggested crops in Buxar based on yield performance:")
    top_yields, forecasts = forecaster.suggest_optimal_crops(district='Buxar')
    if isinstance(top_yields, pd.Series) and not top_yields.empty:
        for crop, yield_value in top_yields.items():
            print(f"- {crop}: {yield_value:.2f} tonnes/hectare")
    else:
        print("No yield data available for suggestions.")
    
    # Generate API response example
    print(f"\nAPI response example for {top_crops.index[0]}:")
    api_response = forecaster.forecasting_api_response(top_crops.index[0], district='Buxar')
    
    # Save API response to file
    with open(os.path.join(output_dir, f"{top_crops.index[0]}_api_response.json"), 'w') as f:
        f.write(str(api_response))
    
    print(f"Status: {api_response['status']}")
    if 'forecast' in api_response and api_response['forecast']:
        print(f"Forecast years: {len(api_response['forecast'])}")
    
    return forecaster

if __name__ == "__main__":
    # Run the Bihar forecasting demo
    forecaster = run_bihar_forecasting_demo() 