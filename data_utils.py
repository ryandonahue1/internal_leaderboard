import pandas as pd
import numpy as np
from datetime import datetime
import math
import json
import os

# MLflow integration will be defined directly in this file

def save_data_to_json(df, filename='leaderboard_data.json'):
    """Save DataFrame to JSON file for deployment fallback."""
    try:
        # Convert DataFrame to JSON-serializable format
        data_dict = {
            'data': df.to_dict('records'),
            'last_updated': datetime.now().isoformat(),
            'total_records': len(df)
        }
        
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2, default=str)
        
        print(f"Successfully saved {len(df)} records to {filename}")
        return True
    except Exception as e:
        print(f"Error saving data to JSON: {e}")
        return False

def load_data_from_json(filename='leaderboard_data.json'):
    """Load data from JSON file as fallback when MLflow is unavailable."""
    try:
        if not os.path.exists(filename):
            print(f"No saved data file found at {filename}")
            return pd.DataFrame(columns=['timestamp', 'task_category', 'model_a', 'model_b', 'winner'])
        
        with open(filename, 'r') as f:
            data_dict = json.load(f)
        
        df = pd.DataFrame(data_dict['data'])
        
        # Ensure timestamp is datetime type
        if 'timestamp' in df.columns and not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        last_updated = data_dict.get('last_updated', 'Unknown')
        print(f"Successfully loaded {len(df)} records from {filename}")
        print(f"Data last updated: {last_updated}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data from JSON: {e}")
        return pd.DataFrame(columns=['timestamp', 'task_category', 'model_a', 'model_b', 'winner'])

def update_saved_data():
    """Update the saved data file with fresh MLflow data. Use this locally to refresh deployment data."""
    try:
        print("Fetching fresh data from MLflow...")
        df = load_data_from_mlflow()
        
        if df.empty:
            print("No MLflow data available to save.")
            return False
        
        success = save_data_to_json(df)
        if success:
            print("✅ Saved data file updated successfully!")
            print("💡 Commit and push the updated leaderboard_data.json to deploy fresh data.")
            return True
        else:
            print("❌ Failed to update saved data file.")
            return False
            
    except Exception as e:
        print(f"Error updating saved data: {e}")
        return False

def load_data_from_mlflow():
    """Load comparison data from MLflow experiments."""
    try:
        import mlflow
        
        # MLflow Configuration
        MLFLOW_TRACKING_URI = "https://mlflow-tracking-api.vlex.io"
        MLFLOW_USER = "ryan.donahue@vlex.com"
        TASK_CATEGORY_MAPPING = {
            "extract_dramatis": "Extract Dramatis",
            "extract_claims": "Extract Claims",
            "summarize_relief": "Summarize Relief"
        }
        
        def clean_model_name(model_name):
            if not model_name:
                return "Unknown"
            name_mappings = {
                "gpt-4.1-mini-2025-04-14": "GPT-4.1-Mini",
                "us.anthropic.claude-3-7-sonnet-20250219-v1:0": "Claude-3.7-Sonnet",
                "gpt-4.1-2025-04-14": "GPT-4.1",
                "anthropic.claude-3-5-sonnet-20240620-v1:0": "Claude-3.5-Sonnet"
            }
            return name_mappings.get(model_name, model_name.replace("us.anthropic.", "").replace("-v1:0", ""))
        
        # Setup MLflow connection
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Set up authentication - try multiple methods
        #mlflow.login(MLFLOW_USER)
        
        # Search for experiments
        experiment_names = ["LLM Judge Complaint Analysis Evals"]
        experiment_ids = []
        all_experiments = mlflow.search_experiments()
        for exp in all_experiments:
            if exp.name in experiment_names or exp.name.startswith("llm_judge_evals"):
                experiment_ids.append(exp.experiment_id)
                print(f"Found experiment: {exp.name} (ID: {exp.experiment_id})")
        
        if not experiment_ids:
            experiment_ids = ["17"]
        
        # Search for runs
        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"]
        )
        
        if runs.empty:
            return pd.DataFrame(columns=['timestamp', 'task_category', 'model_a', 'model_b', 'winner'])
        
        # Transform to comparison format
        comparison_data = []
        for _, run in runs.iterrows():
            try:
                task_plan = run.get('params.task_plan_name', '')
                if task_plan not in TASK_CATEGORY_MAPPING:
                    continue
                
                model_a = clean_model_name(run.get('params.model_a', ''))
                model_b = clean_model_name(run.get('params.model_b', ''))
                if not model_a or not model_b:
                    continue
                
                timestamp = pd.to_datetime(run.get('start_time', datetime.now()))
                task_category = TASK_CATEGORY_MAPPING[task_plan]
                
                # Create individual match records from win counts
                model_a_wins = int(run.get('metrics.model_a_wins', 0))
                model_b_wins = int(run.get('metrics.model_b_wins', 0))
                
                # Create a realistic sequence of winners instead of grouping all A wins first
                winners = [model_a] * model_a_wins + [model_b] * model_b_wins
                
                # Shuffle to create realistic chronological order (use run ID for consistent seed)
                import random
                random.Random(run.get('run_id', 'default')).shuffle(winners)
                
                # Create individual match records with interleaved winners
                for i, winner in enumerate(winners):
                    comparison_data.append({
                        'timestamp': timestamp + pd.Timedelta(minutes=i),
                        'task_category': task_category,
                        'model_a': model_a,
                        'model_b': model_b,
                        'winner': winner
                    })
                
            except Exception as e:
                print(f"Error processing run: {e}")
                continue
        
        if not comparison_data:
            return pd.DataFrame(columns=['timestamp', 'task_category', 'model_a', 'model_b', 'winner'])
        
        df = pd.DataFrame(comparison_data)
        print(f"Successfully created {len(df)} individual match records")
        return df
        
    except Exception as e:
        print(f"Error loading data from MLflow: {e}")
        return pd.DataFrame(columns=['timestamp', 'task_category', 'model_a', 'model_b', 'winner'])

def load_data(use_mlflow=True):
    """Load and preprocess the match data from MLflow with JSON fallback."""
    if not use_mlflow:
        print("Loading from saved data file...")
        return load_data_from_json()
        
    try:
        print("Loading data from MLflow...")
        df = load_data_from_mlflow()
        
        if df.empty:
            print("No MLflow data found, trying saved data file...")
            return load_data_from_json()
        else:
            print(f"Successfully loaded {len(df)} records from MLflow")
            # Ensure timestamp is datetime type
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Auto-save the fresh MLflow data for deployment fallback
            save_data_to_json(df)
            
            return df
            
    except Exception as e:
        print(f"MLflow loading failed: {e}")
        print("Falling back to saved data file...")
        return load_data_from_json()


def calculate_wilson_confidence_interval(wins, total_matches, confidence_level=0.95):
    """
    Calculate Wilson score confidence interval for win rate.
    
    This is more accurate than normal approximation, especially for small samples
    or extreme win rates.
    
    Args:
        wins: Number of wins
        total_matches: Total number of matches
        confidence_level: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        tuple: (lower_bound, upper_bound) as percentages
    """
    if total_matches == 0:
        return (0.0, 100.0)
    
    # Convert confidence level to z-score
    alpha = 1 - confidence_level
    z = 1.96 if confidence_level == 0.95 else abs(np.percentile(np.random.normal(0, 1, 10000), 100 * alpha / 2))
    
    p = wins / total_matches  # Observed proportion
    n = total_matches
    
    # Wilson score interval
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    
    lower = max(0, center - margin)  # Ensure non-negative
    upper = min(1, center + margin)  # Ensure not > 100%
    
    return (round(lower * 100, 1), round(upper * 100, 1))

def get_all_models(df):
    """Get a list of all unique models in the dataset."""
    if df.empty:
        return []
    
    models_a = set(df['model_a'].unique())
    models_b = set(df['model_b'].unique())
    all_models = models_a.union(models_b)
    return sorted(list(all_models))

def calculate_win_rates(df, category=None):
    """Calculate win rates for all models, optionally filtered by category."""
    if category:
        df = df[df['task_category'] == category]
    
    models = get_all_models(df)
    stats = []
    
    for model in models:
        # Count wins
        wins = len(df[df['winner'] == model])
        
        # Count total matches (as model_a or model_b)
        total_matches = len(df[(df['model_a'] == model) | (df['model_b'] == model)])
        
        # Calculate win rate
        win_rate = (wins / total_matches * 100) if total_matches > 0 else 0
        
        # Calculate confidence interval for win rate
        ci_lower, ci_upper = calculate_wilson_confidence_interval(wins, total_matches)
        
        # Calculate asymmetric margins for proper +/- format
        plus_margin = ci_upper - win_rate
        minus_margin = win_rate - ci_lower
        
        # Create asymmetric CI display
        if abs(plus_margin - minus_margin) < 0.1:  # Nearly symmetric
            avg_margin = (plus_margin + minus_margin) / 2
            ci_display = f"± {avg_margin:.1f}%"
        else:  # Asymmetric - show both
            ci_display = f"+{plus_margin:.1f}%/-{minus_margin:.1f}%"
        
        stats.append({
            'model': model,
            'wins': wins,
            'total_matches': total_matches,
            'win_rate': round(win_rate, 1),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'plus_margin': round(plus_margin, 1),
            'minus_margin': round(minus_margin, 1),
            'ci_range': ci_display
        })
    
    # Sort by win rate descending
    stats = sorted(stats, key=lambda x: x['win_rate'], reverse=True)
    return stats

def calculate_elo_ratings(df, category=None, k_factor=32, initial_rating=1200):
    """Calculate ELO ratings for all models."""
    if category:
        df = df[df['task_category'] == category]
    
    models = get_all_models(df)
    
    # Initialize ELO ratings
    elo_ratings = {model: initial_rating for model in models}
    
    # Sort matches by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # Process each match
    for _, match in df_sorted.iterrows():
        model_a = match['model_a']
        model_b = match['model_b']
        winner = match['winner']
        
        # Current ratings
        rating_a = elo_ratings[model_a]
        rating_b = elo_ratings[model_b]
        
        # Expected scores
        expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
        expected_b = 1 / (1 + 10**((rating_a - rating_b) / 400))
        
        # Actual scores
        if winner == model_a:
            score_a, score_b = 1, 0
        elif winner == model_b:
            score_a, score_b = 0, 1
        else:  # This shouldn't happen based on your clarification, but just in case
            score_a, score_b = 0.5, 0.5
        
        # Update ratings
        elo_ratings[model_a] = rating_a + k_factor * (score_a - expected_a)
        elo_ratings[model_b] = rating_b + k_factor * (score_b - expected_b)
    
    # Round ratings
    for model in elo_ratings:
        elo_ratings[model] = round(elo_ratings[model])
    
    return elo_ratings

def create_leaderboard(df, category=None):
    """Create a comprehensive leaderboard combining win rates and ELO."""
    win_stats = calculate_win_rates(df, category)
    elo_ratings = calculate_elo_ratings(df, category)
    
    # Combine data
    leaderboard = []
    for i, stats in enumerate(win_stats):
        model = stats['model']
        leaderboard.append({
            'rank': i + 1,
            'model': model,
            'elo_rating': elo_ratings[model],
            'win_rate': stats['win_rate'],
            'ci_lower': stats['ci_lower'],
            'ci_upper': stats['ci_upper'],
            'plus_margin': stats['plus_margin'],
            'minus_margin': stats['minus_margin'],
            'ci_range': stats['ci_range'],
            'wins': stats['wins'],
            'total_matches': stats['total_matches']
        })
    
    return leaderboard

def create_mini_leaderboard(df, category=None):
    """Create a comprehensive leaderboard combining win rates and ELO."""
    win_stats = calculate_win_rates(df, category)
    
    # Combine data
    leaderboard = []
    for i, stats in enumerate(win_stats):
        model = stats['model']
        leaderboard.append({
            'rank': i + 1,
            'model': model,
            'win_rate': stats['win_rate'],
            'plus_margin': stats['plus_margin'],
            'minus_margin': stats['minus_margin'],
            'ci_range': stats['ci_range']
        })
    
    return leaderboard

def get_top_models_by_category(df, top_n=3):
    """Get top N models for each task category."""
    if df.empty:
        return {}
    
    categories = df['task_category'].unique()
    top_models = {}
    
    for category in categories:
        leaderboard = create_leaderboard(df, category)
        top_models[category] = leaderboard[:top_n]
    
    return top_models

def get_summary_stats(df):
    """Get overall summary statistics."""
    if df.empty:
        return {
            'total_matches': 0,
            'unique_models': 0,
            'categories': [],
            'date_range': 'No data'
        }
    
    total_matches = len(df)
    unique_models = len(get_all_models(df))
    categories = list(df['task_category'].unique())
    
    # Handle timestamp formatting safely
    try:
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        
        # Check if timestamps are valid (not NaT)
        if pd.isna(min_time) or pd.isna(max_time):
            date_range = 'No valid dates'
        else:
            date_range = f"{min_time.strftime('%Y-%m-%d')} to {max_time.strftime('%Y-%m-%d')}"
    except (AttributeError, ValueError):
        date_range = 'Invalid date format'
    
    return {
        'total_matches': total_matches,
        'unique_models': unique_models,
        'categories': categories,
        'date_range': date_range
    } 