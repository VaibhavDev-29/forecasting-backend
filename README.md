# FairChain - AI-Powered Transparent Agri-Marketplace

FairChain is an AI-powered platform that connects farmers, intermediaries, and consumers, ensuring transparent pricing and fair profit distribution in the agricultural supply chain.

## Key Features

1. **AI-Powered Forecasting**: Uses time series models (Prophet, SARIMA, XGBoost, LSTM) to predict crop production and yields
2. **Crop Recommendations**: Suggests optimal crops based on yield performance and market demand
3. **Transparent Pricing**: Provides market price information and forecasts to help farmers make informed decisions
4. **Crop Calendar**: Offers planting and harvesting schedule information for different crops

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- Pandas, NumPy, Matplotlib
- Prophet, Statsmodels, XGBoost, TensorFlow
- Uvicorn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fairchain.git
cd fairchain
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
python api.py
```

The API will be available at http://localhost:8000.

## API Documentation

After starting the API server, you can access the interactive API documentation at http://localhost:8000/docs.

### Finding Available Crops

Before making forecasts or getting recommendations, you should check which crops are available in the system:

```
GET /crops
```

**Response:**
```json
{
  "status": "success",
  "crops": ["Arhar/Tur", "Bajra", "Barley", "Gram", "Jowar", "Maize", "Masoor", "Potato", "Rice", "Wheat"],
  "count": 10
}
```

Use these exact crop names when making requests to other endpoints.

### Main Endpoints

#### Crop Forecasting

```
POST /forecast/crop
```

Generate production or yield forecasts for a specific crop using ensemble AI models.

**Request:**
```json
{
  "crop_name": "Rice",
  "metric": "Production",
  "periods": 5
}
```

**Response:**
```json
{
  "status": "success",
  "crop": "Rice",
  "metric": "Production",
  "historical_data": [
    {
      "year": 2018,
      "Production": 254525.0,
      "area": 78492.0,
      "yield": 3.24
    },
    ...
  ],
  "forecast": [
    {
      "year": 2023,
      "production": 308245.0
    },
    ...
  ],
  "last_updated": "2023-08-01"
}
```

#### Crop Recommendations

```
POST /recommend/optimal-crops
```

Recommend optimal crops to grow based on yield performance and AI forecasts.

**Request:**
```json
{
  "region": "Buxar",
  "top_n": 5
}
```

**Response:**
```json
{
  "status": "success",
  "region": "Buxar",
  "optimal_crops": [
    {
      "crop": "Potato",
      "current_yield": 16.78,
      "yield_trend": "increasing",
      "profit_potential": "high",
      "forecasted_yield": 18.5
    },
    {
      "crop": "Onion",
      "current_yield": 12.0,
      "yield_trend": "stable",
      "profit_potential": "high"
    },
    ...
  ],
  "explanation": "Recommendations based on historical yield performance and AI forecasts",
  "last_updated": "2023-08-01"
}
```

#### Market Prices

```
GET /market-prices/{crop_name}
```

Get current market prices and forecasts for a specific crop.

**Response:**
```json
{
  "status": "success",
  "crop": "Rice",
  "current_price": 3600.45,
  "price_trend": "increasing",
  "price_forecast": [
    {
      "date": "2023-09-01",
      "price": 3720.75
    },
    ...
  ],
  "last_updated": "2023-08-01"
}
```

#### Regional Demand Forecasting

```
GET /forecast/demand/{region}
```

Get demand forecasts for top crops in a specific region.

**Response:**
```json
{
  "status": "success",
  "region": "Buxar",
  "forecasts": [
    {
      "crop": "Rice",
      "forecast": [
        {
          "year": 2023,
          "demand": 328045.25
        },
        ...
      ]
    },
    {
      "crop": "Wheat",
      "forecast": [
        {
          "year": 2023,
          "demand": 295782.4
        },
        ...
      ]
    },
    ...
  ],
  "last_updated": "2023-08-01"
}
```

#### Crop Calendar

```
POST /crop-calendar
```

Get planting and harvesting information for crops.

**Request:**
```json
{
  "crop_name": "Rice"
}
```

**Response:**
```json
{
  "status": "success",
  "crop_calendars": {
    "Rice": [
      {
        "season": "Kharif",
        "planting_time": "June-July",
        "harvesting_time": "October-November"
      },
      {
        "season": "Rabi",
        "planting_time": "October-November",
        "harvesting_time": "March-April"
      }
    ]
  },
  "last_updated": "2023-08-01"
}
```

## Integrating With Your System

The API is designed to be easily integrated with your frontend or mobile applications. Here's how to use it with different frameworks:

### React/Next.js

```javascript
// Example of fetching crop recommendations
const fetchCropRecommendations = async () => {
  try {
    const response = await fetch('http://localhost:8000/recommend/optimal-crops', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        region: 'Buxar',
        top_n: 5
      }),
    });
    
    const data = await response.json();
    if (data.status === 'success') {
      setCropRecommendations(data.optimal_crops);
    }
  } catch (error) {
    console.error('Error fetching crop recommendations:', error);
  }
};
```

### Mobile (React Native)

```javascript
// Example of fetching crop forecasts
const getCropForecast = async (cropName) => {
  try {
    // First get valid crop names
    const cropsResponse = await fetch('http://localhost:8000/crops');
    const cropsData = await cropsResponse.json();
    
    // Validate crop name exists before making forecast request
    if (cropsData.status === 'success' && cropsData.crops.includes(cropName)) {
      const response = await fetch('http://localhost:8000/forecast/crop', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          crop_name: cropName,
          metric: 'Production',
          periods: 5
        }),
      });
      
      const result = await response.json();
      return result;
    } else {
      console.error('Invalid crop name:', cropName);
      return null;
    }
  } catch (error) {
    console.error('Error fetching forecast:', error);
    return null;
  }
};
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Food and Agricultural Organization (FAO) for agricultural data
- Open Weather Map for weather data
- Farmer organizations for domain expertise and feedback 