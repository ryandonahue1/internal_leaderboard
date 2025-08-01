#!/usr/bin/env python3
"""
Script to update the saved leaderboard data from MLflow.
Run this locally to refresh the data file for deployment.

Usage:
    python update_data.py
"""

from data_utils import update_saved_data

if __name__ == "__main__":
    print("🔄 Updating leaderboard data from MLflow...")
    print("=" * 50)
    
    success = update_saved_data()
    
    print("=" * 50)
    if success:
        print("✅ Data update completed successfully!")
        print("📤 Next steps:")
        print("   1. git add leaderboard_data.json")
        print("   2. git commit -m 'Update leaderboard data'")
        print("   3. git push")
        print("   4. Your Streamlit deployment will use the fresh data!")
    else:
        print("❌ Data update failed. Check your MLflow connection.") 