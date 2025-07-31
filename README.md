# 🏆 Internal LLM Leaderboard

A Streamlit-based web application for tracking and comparing AI model performance across different task categories using MLflow experiment data.

## 📋 Overview

This leaderboard tracks AI model performance across three key legal document analysis tasks:
- **Extract Dramatis** 🎭 - Extracting key parties from legal documents
- **Extract Claims** 📋 - Identifying claims from legal documents  
- **Summarize Relief** 📝 - Summarizing requested relief from legal documents

The application connects to your MLflow tracking server to automatically pull experiment results and display comprehensive performance analytics including win rates, ELO ratings, and confidence intervals.

## ✨ Features

- **Real-time MLflow Integration** - Automatically syncs with your MLflow experiments
- **Multiple Task Categories** - Compare models across different evaluation tasks
- **Advanced Statistics** - Wilson confidence intervals and ELO rating system
- **Modern UI** - Beautiful Streamlit interface with responsive design
- **Performance Metrics** - Win rates, confidence intervals, match history
- **Custom MLflow Plugin** - Includes authentication header plugin for secure MLflow access

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Access to MLflow tracking server
- MLflow experiments with evaluation data

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd internal_leaderboard
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install MLflow header plugin**
   ```bash
   cd mlflow_header_plugin
   pip install -e .
   cd ..
   ```

### Configuration

Update the MLflow configuration in `data_utils.py`:

```python
# MLflow Configuration
MLFLOW_TRACKING_URI = "https://your-mlflow-server.com"
MLFLOW_USER = "your.email@company.com"
```

### Running the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📊 Data Format

The application expects MLflow experiments with the following structure:

### Required Parameters
- `task_plan_name`: One of `extract_dramatis`, `extract_claims`, `summarize_relief`
- `model_a`: Name of the first model being compared
- `model_b`: Name of the second model being compared

### Required Metrics
- `model_a_wins`: Number of wins for model A
- `model_b_wins`: Number of wins for model B

### Example MLflow Run
```python
with mlflow.start_run():
    mlflow.log_param("task_plan_name", "extract_dramatis")
    mlflow.log_param("model_a", "gpt-4.1-mini-2025-04-14")
    mlflow.log_param("model_b", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    mlflow.log_metric("model_a_wins", 15)
    mlflow.log_metric("model_b_wins", 10)
```

## 🔧 Components

### Core Files

- **`app.py`** - Main Streamlit application with UI components
- **`data_utils.py`** - Data processing, MLflow integration, and statistical calculations
- **`requirements.txt`** - Python dependencies

### MLflow Plugin

- **`mlflow_header_plugin/`** - Custom MLflow plugin for request header authentication
  - Provides secure authentication for MLflow API requests
  - Automatically installed during setup

## 📈 Statistical Methods

### Win Rate Calculation
- Simple win percentage: `wins / total_matches * 100`
- Uses Wilson Score Confidence Intervals (more accurate than normal approximation)
- Displays asymmetric confidence intervals when appropriate

### ELO Rating System
- Initial rating: 1200
- K-factor: 32
- Updates chronologically based on match timestamps
- Provides relative skill assessment independent of win rate

## 🎨 UI Features

### Overview Page
- Summary statistics across all categories
- Mini leaderboards for each task category
- Overall performance rankings

### Category Detail Pages
- Detailed rankings for specific task categories
- Recent match history
- Category-specific performance metrics

### Modern Design
- Gradient card layouts
- Responsive design
- Interactive navigation
- Real-time data refresh

## 🔄 Data Refresh

The application automatically caches data for 5 minutes for performance. To manually refresh:

1. Use the "Refresh Data" button in the sidebar
2. Or restart the Streamlit application

## 🛠️ Development

### Project Structure
```
internal_leaderboard/
├── app.py                      # Main Streamlit app
├── data_utils.py              # Data processing & MLflow integration
├── requirements.txt           # Dependencies
├── mlflow_header_plugin/      # Custom MLflow plugin
│   ├── mlflow_header_plugin/
│   │   ├── __init__.py
│   │   └── request_header_provider.py
│   └── setup.py
└── README.md                  # This file
```

### Adding New Task Categories

1. Update `TASK_CATEGORY_MAPPING` in `data_utils.py`
2. Add category-specific styling in `app.py` (optional)
3. Ensure MLflow experiments use the new task category name

### Customizing Model Name Display

Update the `clean_model_name()` function in `data_utils.py` to add new model name mappings.

## 🔐 Security

- MLflow authentication handled via custom header plugin
- No sensitive data stored in code
- Environment-based configuration recommended for production

## 📝 License

[Your License Here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues or questions:
- Create an issue in the repository
- Contact the development team
- Check MLflow connection and experiment data format

---

**Built with Streamlit** • **Powered by MLflow** 