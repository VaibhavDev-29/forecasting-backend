from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uvicorn
import os
import json
from main import CropForecastingModule
from bihar_predictions import BiharForecastingModule

# Initialize FastAPI app
app = FastAPI(
    title="FairChain - AI-Powered Forecasting API",
    description="Provides crop forecasting and recommendations for the FairChain platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize forecasting module
forecasting_module = None

# Data models for API requests and responses
class CropForecastRequest(BaseModel):
    crop_name: Optional[str] = "wheat"
    region: Optional[str] = "Buxar"
    metric: str = "Production"
    top_n: int = Field(5, ge=1, le=20, description="Number of top crops to recommend")
    periods: int = Field(5, ge=1, le=10, description="Number of periods (years) to forecast")

class YieldAnalysisRequest(BaseModel):
    region: str = "Buxar"
    top_n: int = Field(5, ge=1, le=20, description="Number of top crops to recommend")

class CropCalendarRequest(BaseModel):
    crop_name: Optional[str] = None

class ForecastResponse(BaseModel):
    status: str
    crop: str
    metric: str
    historical_data: List[Dict[str, Any]]
    forecast: List[Dict[str, Any]]
    forecast_confidence: Optional[float] = None
    last_updated: str

class OptimalCropsResponse(BaseModel):
    status: str
    region: str
    optimal_crops: List[Dict[str, Any]]
    explanation: str
    last_updated: str

class CropCalendarResponse(BaseModel):
    status: str
    crop_calendars: Dict[str, List[Dict[str, str]]]
    last_updated: str

class MarketPriceResponse(BaseModel):
    status: str
    crop: str
    current_price: float
    price_trend: str
    price_forecast: List[Dict[str, Any]]
    last_updated: str

class CropForecastResponse(BaseModel):
    crop: str
    forecast_values: List[float]
    forecast_years: List[int]

# Dependency to get the forecasting module
async def get_forecasting_module():
    global forecasting_module
    if forecasting_module is None:
        try:
            # Check if the files exist
            import os
            bihar_path = "bihar-dataset.csv"
            fallback_path = "dataset.csv"
            
            if os.path.exists(bihar_path):
                print(f"Found bihar-dataset.csv, size: {os.path.getsize(bihar_path)/1024:.2f} KB")
                data_path = bihar_path
            elif os.path.exists(fallback_path):
                print(f"bihar-dataset.csv not found, using fallback dataset.csv, size: {os.path.getsize(fallback_path)/1024:.2f} KB")
                data_path = fallback_path
            else:
                print("Warning: Neither bihar-dataset.csv nor dataset.csv found!")
                data_path = bihar_path  # Use bihar path and let it fail with proper error
            
            # Initialize with explicit bihar-dataset.csv
            forecasting_module = BiharForecastingModule(data_path=data_path)
            forecasting_module.preprocess_data()
            
            # Verify data was loaded
            if forecasting_module.processed_data is not None:
                print(f"Successfully initialized forecasting module with {data_path}")
                print(f"Processed {len(forecasting_module.processed_data)} data points")
                districts = forecasting_module.processed_data.get('District', pd.Series()).unique()
                print(f"Found {len(districts)} districts: {', '.join(str(d) for d in districts if d is not None)}")
            else:
                print("Warning: processed_data is None after initialization!")
        except Exception as e:
            print(f"Failed to initialize forecasting module: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize forecasting module: {str(e)}"
            )
    return forecasting_module

# Endpoints
@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Welcome to FairChain AI Forecasting API",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs"
    }

@app.post("/forecast/crop", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast_crop(
    request: CropForecastRequest,
    forecaster: CropForecastingModule = Depends(get_forecasting_module)
):
    """
    Generate production or yield forecast for a specific crop.
    
    This endpoint provides predictions for future crop production or yield based on
    historical data and advanced time series models (Prophet, SARIMA, XGBoost, LSTM).
    """
    response = forecaster.forecasting_api_response(
        crop_name=request.crop_name,
        metric=request.metric
    )
    
    if response["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=response["message"]
        )
    
    return response

@app.post("/recommend/optimal-crops", response_model=OptimalCropsResponse, tags=["Recommendations"])
async def recommend_optimal_crops(
    request: YieldAnalysisRequest,
    forecaster: CropForecastingModule = Depends(get_forecasting_module)
):
    """
    Recommend optimal crops to grow based on yield performance and forecasts.
    
    This endpoint analyzes historical yield data and uses AI forecasting to suggest 
    the most profitable crops for farmers to grow in the next season.
    """
    try:
        print(f"Received request for optimal crops in region: {request.region}")
        
        # Check if we have valid data
        if forecaster.processed_data is None or len(forecaster.processed_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail="No processed data available. Please check your dataset."
            )
            
        # Get available crops for debugging
        available_crops = sorted(forecaster.processed_data['Crop'].unique().tolist())
        print(f"Available crops: {', '.join(available_crops[:5])}...")
        
        # Handle different parameter names based on forecaster type
        if isinstance(forecaster, BiharForecastingModule):
            print(f"Using BiharForecastingModule with district={request.region}")
            # This method has been fixed to properly handle district matching
            top_yields, forecasts = forecaster.suggest_optimal_crops(
                district=request.region,
                top_n=request.top_n,
                optimization_method='model_based'  # Use the new optimization approach
            )
        else:
            print(f"Using CropForecastingModule with region={request.region}")
            top_yields, forecasts = forecaster.suggest_optimal_crops(
                region=request.region,
                top_n=request.top_n
            )
        
        # Validate we got valid results
        if top_yields.empty:
            print(f"No yields found for region: {request.region}")
            return {
                "status": "warning",
                "region": request.region,
                "optimal_crops": [],
                "explanation": f"No suitable crops found for your region based on forecasting models.",
                "last_updated": datetime.now().strftime('%Y-%m-%d')
            }
        
        # Process recommendations
        optimal_crops = []
        
        print(f"Found {len(top_yields)} top yielding crops: {', '.join(top_yields.index)}")
        
        for crop, yield_value in top_yields.items():
            try:
                # Process crop information
                crop_info = {
                    "crop": crop,
                    "current_yield": float(yield_value),
                    "yield_trend": "stable"
                }
                
                # Add forecast data if available
                if crop in forecasts:
                    forecast_data = forecasts[crop]
                    
                    # Handle different forecast formats (BiharForecastingModule vs CropForecastingModule)
                    if 'trend' in forecast_data:
                        # Map trend to API format
                        trend_map = {
                            "growing": "increasing",
                            "declining": "decreasing",
                            "stable": "stable"
                        }
                        crop_info["yield_trend"] = trend_map.get(forecast_data['trend'], "stable")
                    
                    # Add growth potential
                    if 'growth_potential' in forecast_data:
                        crop_info["growth_potential"] = float(forecast_data['growth_potential'])
                    
                    # Add confidence score
                    if 'confidence_score' in forecast_data:
                        crop_info["confidence_score"] = float(forecast_data['confidence_score'])
                    
                    # Get forecasted yield
                    if 'forecast_values' in forecast_data and forecast_data['forecast_values']:
                        # BiharForecastingModule format with separate values and years
                        crop_info["forecasted_yield"] = float(forecast_data['forecast_values'][-1])
                    elif 'forecast' in forecast_data:
                        # Handle forecast format from CropForecastingModule
                        if isinstance(forecast_data['forecast'], list):
                            # List of dicts with 'forecast' key
                            points = forecast_data['forecast']
                            if points and 'forecast' in points[-1]:
                                crop_info["forecasted_yield"] = float(points[-1]['forecast'])
                    
                    # Determine profit potential
                    if 'profit_potential' in forecast_data:
                        profit_score = forecast_data['profit_potential']
                        # Convert numeric score to text description
                        if isinstance(profit_score, (int, float)):
                            if profit_score > 75:
                                crop_info["profit_potential"] = "high"
                            elif profit_score > 40:
                                crop_info["profit_potential"] = "moderate"
                            else:
                                crop_info["profit_potential"] = "low"
                        else:
                            crop_info["profit_potential"] = "moderate"
                    elif 'growth_potential' in forecast_data:
                        # Derive profit potential from growth
                        growth = forecast_data['growth_potential']
                        if growth > 15:
                            crop_info["profit_potential"] = "high"
                        elif growth > 5:
                            crop_info["profit_potential"] = "moderate"
                        else:
                            crop_info["profit_potential"] = "low"
                    else:
                        crop_info["profit_potential"] = "moderate"
                else:
                    # Default values if no forecast data
                    crop_info["profit_potential"] = "moderate"
                    
                optimal_crops.append(crop_info)
            except Exception as e:
                print(f"Error processing crop {crop}: {str(e)}")
                # Still include the crop with basic info
                optimal_crops.append({
                    "crop": crop,
                    "current_yield": float(yield_value),
                    "yield_trend": "stable",
                    "profit_potential": "moderate"
                })
        
        # Final check to ensure we have recommendations
        if not optimal_crops:
            print("No valid crops after processing")
            return {
                "status": "warning",
                "region": request.region,
                "optimal_crops": [],
                "explanation": f"No suitable crops could be analyzed for {request.region}.",
                "last_updated": datetime.now().strftime('%Y-%m-%d')
            }
        
        print(f"Returning {len(optimal_crops)} recommended crops based on AI forecasting models")
        
        return {
            "status": "success",
            "region": request.region,
            "optimal_crops": optimal_crops,
            "explanation": f"Recommendations based on AI forecasting models that analyzed historical yield data and projected future performance for {request.region}",
            "last_updated": datetime.now().strftime('%Y-%m-%d')
        }
    except Exception as e:
        # Log the error for debugging
        print(f"Error in recommend_optimal_crops: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to return a simple response with basic crop data
        try:
            # Get basic crop data without forecasting
            basic_data = forecaster.processed_data
            metric = 'Yield'
            
            if basic_data is not None and len(basic_data) > 0:
                # Ensure numeric
                basic_data[metric] = pd.to_numeric(basic_data[metric], errors='coerce')
                
                # Get top crops
                top_basic_crops = basic_data.groupby('Crop')[metric].mean().nlargest(request.top_n)
                
                if not top_basic_crops.empty:
                    basic_recommendations = []
                    for crop, value in top_basic_crops.items():
                        if not any(term in str(crop).lower() for term in ['total', 'district']):
                            basic_recommendations.append({
                                "crop": crop,
                                "current_yield": float(value),
                                "yield_trend": "stable",
                                "profit_potential": "moderate"
                            })
                    
                    if basic_recommendations:
                        return {
                            "status": "partial",
                            "region": request.region,
                            "optimal_crops": basic_recommendations,
                            "explanation": f"Limited recommendations based on historical data only (forecasting failed).",
                            "last_updated": datetime.now().strftime('%Y-%m-%d')
                        }
        except Exception as fallback_error:
            print(f"Fallback recommendation also failed: {str(fallback_error)}")
        
        # If all else fails, raise the original exception
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate crop recommendations: {str(e)}"
        )

@app.post("/crop-calendar", response_model=CropCalendarResponse, tags=["Crop Information"])
async def get_crop_calendar(
    request: CropCalendarRequest,
    forecaster: CropForecastingModule = Depends(get_forecasting_module)
):
    """
    Get planting and harvesting calendar for crops.
       
    This endpoint provides seasonal information about when to plant and harvest specific crops,
    helping farmers with planning and scheduling activities.
    """
    try:
        calendar = forecaster.generate_crop_calendar()
        
        # Filter for specific crop if provided
        if request.crop_name:
            if request.crop_name in calendar:
                calendar = {request.crop_name: calendar[request.crop_name]}
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Crop {request.crop_name} not found in calendar"
                )
        
        return {
            "status": "success",
            "crop_calendars": calendar,
            "last_updated": datetime.now().strftime('%Y-%m-%d')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate crop calendar: {str(e)}"
        )

@app.get("/market-prices/{crop_name}", response_model=MarketPriceResponse, tags=["Market Information"])
async def get_market_prices(
    crop_name: str,
    forecaster: CropForecastingModule = Depends(get_forecasting_module)
):
    """
    Get current market prices and forecasts for a specific crop.
    
    This endpoint provides real-time market prices and future price predictions
    for a specific crop, helping farmers make informed selling decisions.
    """
    # This is a mock implementation - in a real system, this would connect to market price APIs
    try:
        # Get valid crops (excluding district names)
        exclude_terms = ['buxar', 'bhojpur', 'bhagalpur', 'bihar', 'district', 'total']
        all_crops = forecaster.processed_data['Crop'].unique()
        valid_crops = [
            crop for crop in all_crops 
            if not any(exclude in crop.lower() for exclude in exclude_terms)
        ]
        
        # Check if the requested crop exists in our system
        if crop_name not in valid_crops:
            # If not found, check if it's a case sensitivity issue
            matching_crops = [c for c in valid_crops if c.lower() == crop_name.lower()]
            if matching_crops:
                # Use the correct case version
                crop_name = matching_crops[0]
            else:
                # No match at all
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Crop '{crop_name}' not found in the system. Available crops: {', '.join(valid_crops[:10])}..."
                )
            
        # Get production data for this crop to base price estimations on
        crop_data = forecaster.processed_data[forecaster.processed_data['Crop'] == crop_name]
        
        # Mock price calculation based on recent production trends (inverse relationship)
        recent_years = sorted(crop_data['Year'].unique())[-3:]
        recent_data = crop_data[crop_data['Year'].isin(recent_years)]
        
        # Mock current price (would be replaced with actual API data)
        base_price = 2000  # Base price in rupees per quintal
        
        # Adjust price based on crop type (some crops are more valuable)
        crop_price_multipliers = {
            "Rice": 1.8,
            "Wheat": 1.5,
            "Maize": 1.2,
            "Potato": 0.8,
            "Onion": 1.4,
            "Sugarcane": 0.3,
            "Cotton": 4.0,
            "Gram": 3.5,
            "Rapeseed": 3.8,
            "Mustard": 3.8
        }
        
        # Apply crop-specific multiplier if available
        for crop_key, multiplier in crop_price_multipliers.items():
            if crop_key.lower() in crop_name.lower():
                base_price *= multiplier
                break
        
        # Adjust based on recent production trends
        if len(recent_data) > 0:
            recent_trend = recent_data.groupby('Year')['Production'].mean()
            if len(recent_trend) > 1:
                if recent_trend.iloc[-1] > recent_trend.iloc[0]:
                    # Production increasing, so price decreases
                    trend = "decreasing"
                    price_multiplier = 0.9
                else:
                    # Production decreasing, so price increases
                    trend = "increasing"
                    price_multiplier = 1.1
            else:
                trend = "stable"
                price_multiplier = 1.0
        else:
            trend = "unknown"
            price_multiplier = 1.0
            
        current_price = base_price * price_multiplier
        
        # Generate mock price forecast
        today = datetime.now()
        forecast = []
        for i in range(1, 6):
            future_date = today + timedelta(days=i*30)
            # Add some random variation to the price
            forecasted_price = current_price * (1 + (np.random.random() - 0.5) * 0.1)
            forecast.append({
                "date": future_date.strftime('%Y-%m-%d'),
                "price": round(forecasted_price, 2)
            })
        
        return {
            "status": "success",
            "crop": crop_name,
            "current_price": round(current_price, 2),
            "price_trend": trend,
            "price_forecast": forecast,
            "last_updated": datetime.now().strftime('%Y-%m-%d')
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve market prices: {str(e)}"
        )

@app.get("/crops", tags=["Crop Information"])
async def list_available_crops(
    forecaster: CropForecastingModule = Depends(get_forecasting_module)
):
    """
    Get a list of all available crops in the system.
    """
    try:
        if forecaster.processed_data is None or len(forecaster.processed_data) == 0:
            return {
                "status": "error",
                "message": "No crop data available. Please check dataset.",
                "crops": [],
                "count": 0
            }
            
        # Filter out district/region names from crops
        exclude_terms = ['buxar', 'bhojpur', 'bhagalpur', 'bihar', 'district', 'total']
        all_crops = forecaster.processed_data['Crop'].unique()
        
        valid_crops = [
            crop for crop in all_crops 
            if not any(exclude in crop.lower() for exclude in exclude_terms)
        ]
        
        sorted_crops = sorted(valid_crops)
        
        return {
            "status": "success",
            "crops": sorted_crops,
            "count": len(sorted_crops)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve crop list: {str(e)}"
        )

@app.get("/regions", tags=["Regional Information"])
async def list_available_regions(
    forecaster: CropForecastingModule = Depends(get_forecasting_module)
):
    """
    Get a list of all available regions in the system.
    """
    try:
        # Hard-code the districts from Bihar since the extraction is having issues
        bihar_districts = [
            "Araria", "Arwal", "Aurangabad", "Banka", "Begusarai",
            "Bhagalpur", "Bhojpur", "Buxar", "Darbhanga", "East Champaran",
            "Gaya", "Gopalganj", "Jamui", "Jehanabad", "Kaimur",
            "Katihar", "Khagaria", "Kishanganj", "Lakhisarai", "Madhepura",
            "Madhubani", "Munger", "Muzaffarpur", "Nalanda", "Nawada",
            "Patna", "Purnia", "Rohtas", "Saharsa", "Samastipur",
            "Saran", "Sheikhpura", "Sheohar", "Sitamarhi", "Siwan",
            "Supaul", "Vaishali", "West Champaran"
        ]
        
        print(f"Returning {len(bihar_districts)} districts")
        
        return {
            "status": "success",
            "regions": bihar_districts,
            "count": len(bihar_districts)
        }
    except Exception as e:
        print(f"Error in list_available_regions: {str(e)}")
        # Always return at least Buxar as a fallback
        return {
            "status": "error",
            "message": f"Failed to retrieve regions: {str(e)}",
            "regions": ["Buxar"],
            "count": 1
        }

@app.post("/api/forecast/regional_demand", response_model=List[CropForecastResponse])
async def forecast_regional_demand(
    request: CropForecastRequest,
    forecaster: CropForecastingModule = Depends(get_forecasting_module)
):
    """
    Generate crop production forecasts for a specific region
    """
    try:
        # Get regional forecasts
        forecast_results = forecaster.forecast_regional_demand(
            region=request.region, 
            top_n=request.top_n, 
            periods=request.periods
        )
        
        # Format results
        formatted_results = []
        
        if forecast_results:
            for crop, data in forecast_results.items():
                # Skip any district names that got into the results
                if any(district in crop.lower() for district in ['buxar', 'bhojpur', 'bhagalpur', 'bihar', 'district', 'total']):
                    continue
                    
                if 'forecast' in data and 'years' in data:
                    crop_forecast = CropForecastResponse(
                        crop=crop,
                        forecast_values=[float(val) for val in data['forecast']],
                        forecast_years=[int(yr) for yr in data['years']]
                    )
                    formatted_results.append(crop_forecast)
        
        return formatted_results
    except Exception as e:
        print(f"Error in forecast_regional_demand: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 