# 🏆 Internal LLM Leaderboard

A Streamlit web application for tracking and visualizing AI model performance across different task categories within your company's AI assistant experiments.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** to `http://localhost:8501`

## 📊 Features

### Overview Page
- **Summary Statistics**: Total matches, active models, categories, and date range
- **Task Category Cards**: Top 3 performing models per category (Text, WebDev, Vision)
- **Overall Leaderboard**: Combined rankings across all task categories
- **Interactive Navigation**: Click "View All" buttons to dive into specific categories

### Task Detail Pages
- **Category-specific Rankings**: Detailed leaderboard for Text, WebDev, or Vision tasks
- **Recent Match History**: Last 10 matches in each category
- **Performance Metrics**: Win rates, ELO ratings, total matches

### Metrics Calculated
- **Win Rate**: Percentage of matches won by each model
- **ELO Rating**: Chess-style rating system (starts at 1200, ±32 points per match)
- **Match Statistics**: Wins and total matches per model

## 📁 File Structure

```
internal_leaderboard/
├── app.py              # Main Streamlit application
├── data_utils.py       # Data processing and metric calculations
├── mock_data.csv       # Sample match data
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 Data Format

The application expects CSV data with the following columns:
- `timestamp`: Match timestamp (YYYY-MM-DD HH:MM:SS)
- `task_category`: Category (Text, WebDev, Vision, etc.)
- `model_a`: First model in comparison
- `model_b`: Second model in comparison
- `winner`: Winning model name (must match model_a or model_b)

Example:
```csv
timestamp,task_category,model_a,model_b,winner
2025-07-20 10:00:00,Text,Gemini-2.5-Pro,Claude-Opus-4,Gemini-2.5-Pro
```

## 🔗 MLflow Integration

This application now supports live MLflow data! Toggle between mock data and real experiment results.

### **Quick Setup:**
1. **Install MLflow**: `pip install mlflow>=2.0.0`
2. **Test connection**: `python test_mlflow.py`
3. **Run app**: `streamlit run app.py --server.port 8502`
4. **Toggle data source** in the sidebar

### **Requirements:**
- MLflow experiment ID 17 with comparison runs
- Parameters: `task_plan_name`, `model_a`, `model_b`, `evaluation_timestamp`
- Metrics: `model_a_wins`, `model_b_wins`, `total_evaluations`
- Supported tasks: `extract_dramatis`, `extract_claims`

### **Configuration:**
Edit `mlflow_connector.py` to:
- Set your MLflow tracking URI
- Add authentication credentials  
- Map additional model names
- Add new task categories

## 🛠️ Customization

### Adding New Task Categories
Simply include new `task_category` values in your data - the app will automatically detect and create pages for them.

### Modifying ELO Parameters
Edit the `calculate_elo_ratings()` function in `data_utils.py`:
- `k_factor`: Rating change magnitude (default: 32)
- `initial_rating`: Starting rating (default: 1200)

### Styling
Customize the CSS in `app.py` to match your company's branding.

## 📝 License

Internal use only - Company confidential 