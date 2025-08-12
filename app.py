import streamlit as st
import pandas as pd
from data_utils import (
    load_data, create_leaderboard, get_top_models_by_category, 
    get_summary_stats, get_all_models, create_mini_leaderboard  
)

# Page configuration
st.set_page_config(
    page_title="Internal LLM Leaderboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add: simple company login gate using Streamlit OIDC
ALLOWED_EMAIL_DOMAIN = "@vlex.com"

def require_company_login():
    """Enforce login via OIDC and restrict to allowed email domain."""
    # If user is not logged in, show login button and stop.
    if not getattr(st.user, "is_logged_in", False):
        st.title("Sign in required")
        st.info("This app is restricted to vLex employees. Use your Google work account.")
        # Change: call st.login directly to avoid callback edge cases
        if st.button("Log in with Google", use_container_width=True):
            st.login()
            st.stop()
        st.stop()
    
    # After login, validate email domain
    user_email = None
    try:
        # st.user is dict-like; try attribute then key access
        user_email = getattr(st.user, "email", None) or st.user.get("email")
    except Exception:
        user_email = getattr(st.user, "email", None)
    
    if not user_email or not str(user_email).lower().endswith(ALLOWED_EMAIL_DOMAIN):
        st.error("Access restricted to vlex.com accounts. You are logged in as: {}".format(user_email or "unknown"))
        st.button("Log out", on_click=st.logout, use_container_width=True)
        st.stop()

# Custom CSS for modern styling
st.markdown("""
<style>
    .category-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        color: white;
        position: relative;
        overflow: hidden;
    }
    .category-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        z-index: 0;
    }
    .category-content {
        position: relative;
        z-index: 1;
    }
    .category-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .model-row {
        background: rgba(255, 255, 255, 0.15);
        padding: 0.75rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .rank-badge {
        background: linear-gradient(45deg, #ff6b6b, #ffa500);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 2rem;
    }
    .rank-1 { background: linear-gradient(45deg, #ffd700, #ffed4e); color: #333; }
    .rank-2 { background: linear-gradient(45deg, #c0c0c0, #e5e5e5); color: #333; }
    .rank-3 { background: linear-gradient(45deg, #cd7f32, #daa520); color: white; }
    .view-all-btn {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    .view-all-btn:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .nav-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
         .nav-button.active {
         background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
         box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
     }
     .button-container .stButton > button {
         background: rgba(255, 255, 255, 0.2) !important;
         border: 1px solid rgba(255, 255, 255, 0.4) !important;
         color: white !important;
         border-radius: 20px !important;
         font-weight: 600 !important;
         font-size: 0.9rem !important;
         padding: 8px 16px !important;
         transition: all 0.2s ease !important;
         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
         backdrop-filter: blur(5px) !important;
     }
     .button-container .stButton > button:hover {
         background: rgba(255, 255, 255, 0.3) !important;
         transform: translateY(-1px) !important;
         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
     }
</style>
""", unsafe_allow_html=True)

def load_app_data(use_mlflow=True):
    """Load and cache the data."""
    return load_data(use_mlflow=use_mlflow)

@st.cache_data(ttl=300)  # Cache for 5 minutes when using MLflow
def get_cached_data(use_mlflow=True):
    """Cache data loading for better performance."""
    return load_app_data(use_mlflow=use_mlflow)

def get_data_source_info():
    """Determine what data source is being used."""
    import os
    
    # Check if we can import mlflow (indicates MLflow is available)
    try:
        import mlflow
        mlflow_available = True
    except ImportError:
        mlflow_available = False
    
    # Check if saved data file exists
    saved_data_exists = os.path.exists('leaderboard_data.json')
    
    # Determine likely data source
    if mlflow_available:
        return "live", "🔴 Live MLflow Data", "Loading fresh data from MLflow experiments"
    elif saved_data_exists:
        return "saved", "📁 Saved Data File", "Using cached data file (leaderboard_data.json)"
    else:
        return "none", "⚠️ No Data Source", "No MLflow connection or saved data file found"

def render_task_category_card(category, top_models, coming_soon=False):
    """Render a modern task category card with top models."""
    # Category-specific styling and emojis
    category_config = {
        'Extract Dramatis': {'emoji': '🎭', 'icon': '👥'},
        'Extract Claims': {'emoji': '📋', 'icon': '⚖️'},
        'Summarize Relief': {'emoji': '📝', 'icon': '⚡'}
    }
    
    config = category_config.get(category, {'emoji': '📊', 'icon': '🔍'})
    
    # Apply coming soon styling
    if coming_soon:
        card_style = '''
            background: linear-gradient(135deg, #9ca3af 0%, #6b7280 100%);
            opacity: 0.7;
            filter: grayscale(0.3);
        '''
    else:
        if category == 'Extract Dramatis':
            card_style = 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'
        elif category == 'Extract Claims':
            card_style = 'background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'
        elif category == 'Summarize Relief':
            card_style = 'background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'
        else:
            card_style = 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'
    
    st.markdown(f'''
        <div class="category-card" style="{card_style}">
            <div class="category-content">
                <div class="category-title">
                    <span style="font-size: 2rem;">{config['emoji']}</span>
                    <span>{category}</span>
                    <span style="font-size: 1.2rem; opacity: 0.8;">{config['icon']}</span>
                    {f'<span style="font-size: 0.9rem; opacity: 0.7; margin-left: 0.5rem;">• Coming Soon</span>' if coming_soon else ''}
                
    ''', unsafe_allow_html=True)
    
    if coming_soon:
        # Coming soon content
        st.markdown('''
            <div style="text-align: center; padding: 2rem 0;">
                <div style="font-size: 3rem; opacity: 0.6; margin-bottom: 1rem;">🚧</div>
                <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">Coming Soon</div>
                <div style="font-size: 0.9rem; opacity: 0.8;">Vision task evaluations are being prepared</div>
            </div>
        ''', unsafe_allow_html=True)
    else:
        # Top models
        for i, model_data in enumerate(top_models[:3]):
            rank = model_data['rank'] 
            model = model_data['model']
            win_rate = model_data['win_rate']
            matches = model_data['total_matches']
            elo = model_data['elo_rating']
            
            # Rank-specific styling
            rank_class = f"rank-{rank}" if rank <= 3 else "rank-badge"
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""

            
            
            
            st.markdown(f'''
                <div class="model-row">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                            <span class="rank-badge {rank_class}">{medal}#{rank}</span>
                            <span style="font-weight: 600; font-size: 1rem;">{model}</span>
                        </div>
                        <div style="text-align: right; opacity: 0.9;">
                            <div style="font-weight: 700; font-size: 1.1rem;">{win_rate}%</div>
                            <div style="font-size: 0.8rem; opacity: 0.8;">{matches} matches • ELO {elo}</div>
                        </div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        
        # Custom styled View All button container
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("View Detailed Rankings", key=f"btn_{category}", help=f"See full {category} leaderboard"):
                st.session_state.selected_page = category
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)

def render_mini_leaderboard(category, df):
    """Render a mini leaderboard for a specific task category."""
    from data_utils import create_leaderboard
    
    st.markdown(f"**{category}**")
    
    # Create complete leaderboard for this category
    category_leaderboard = create_leaderboard(df, category)
    
    if not category_leaderboard:
        st.info(f"No data available")
        return
    
    # Create a proper DataFrame with the correct columns
    display_df = pd.DataFrame(category_leaderboard)
    
    # Select and rename only the columns we want to show
    mini_df = display_df[['rank', 'model', 'win_rate']].copy()
    mini_df.columns = ['Rank', 'Model', 'Win Rate (%)']
    
    st.dataframe(
        mini_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Win Rate (%)": st.column_config.ProgressColumn(
                "Win Rate (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            )
        }
    )
    
    # Add a "View Details" button with unique key
    if st.button(f"View Full {category} Leaderboard →", key=f"mini_btn_{category}", help=f"See detailed {category} rankings", use_container_width=True):
        st.session_state.selected_page = category
        st.rerun()

def render_overview_page():
    """Render the main overview page."""
    # Always use MLflow data
    use_mlflow = True
    st.session_state.use_mlflow = use_mlflow
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Source")
    
    # Get data source information
    source_type, source_title, source_description = get_data_source_info()
    
    if source_type == "live":
        st.sidebar.success(f"**{source_title}**")
        st.sidebar.info(source_description)
    elif source_type == "saved":
        st.sidebar.warning(f"**{source_title}**")
        st.sidebar.info(source_description)
        
        # Check when the file was last updated
        import os
        if os.path.exists('leaderboard_data.json'):
            try:
                import json
                with open('leaderboard_data.json', 'r') as f:
                    data_info = json.load(f)
                last_updated = data_info.get('last_updated', 'Unknown')
                st.sidebar.caption(f"Last updated: {last_updated}")
            except:
                pass
    else:
        st.sidebar.error(f"**{source_title}**")
        st.sidebar.warning(source_description)
    
    df = get_cached_data(use_mlflow=use_mlflow)
    
    # Check if we have any data
    if df.empty:
        st.warning("No data available. Please check your MLflow connection.")
        st.info("Verify that your MLflow experiments contain evaluation data.")
        return
    
    # Define active categories from MLflow data
    active_categories = []
    if not df.empty:
        # Get actual categories from the data
        available_categories = df['task_category'].unique().tolist()
        active_categories = available_categories
    
    # Define categories we know about from MLflow
    expected_categories = ['Extract Dramatis', 'Extract Claims', 'Summarize Relief']
    coming_soon_categories = []  # No coming soon categories for now
    
    # Header
    st.title("🏆 Internal LLM Leaderboard")
    st.markdown("**Compare AI model performance across different task categories**")
    
    # Summary stats (only active categories)
    active_df = df[df['task_category'].isin(active_categories)]
    
    if active_df.empty:
        st.warning(f"No data found for categories: {', '.join(active_categories)}")
        if use_mlflow:
            available_categories = df['task_category'].unique().tolist()
            st.info(f"Available categories in your data: {', '.join(available_categories)}")
            st.info("Check if your MLflow experiment has runs with the expected task categories.")
        return
    
    stats = get_summary_stats(active_df)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", stats['total_matches'])
    with col2:
        st.metric("Active Models", stats['unique_models'])
    with col3:
        st.metric("Active Tasks", len(active_categories))

    
    st.markdown("---")
    
    # Task Leaderboards section
    if active_categories:
        st.subheader("Task Leaderboards")
        st.markdown("**Performance rankings for each task category**")
        
        top_models_by_category = get_top_models_by_category(active_df)
        
        # Create columns for mini leaderboards
        mini_cols = st.columns(len(active_categories))
        
        for i, category in enumerate(active_categories):
            with mini_cols[i]:
                if category in top_models_by_category:
                    render_mini_leaderboard(category, active_df)



    
    st.markdown("---")
    
    # Overall leaderboard (only active categories)
    st.subheader("Overall Leaderboard")
    overall_leaderboard = create_leaderboard(active_df)
    
    # Check if we have any leaderboard data
    if not overall_leaderboard:
        st.warning("No leaderboard data available.")
        st.info("Please check your data source or try refreshing the data.")
        return
    
    # Create dataframe for display
    display_df = pd.DataFrame(overall_leaderboard)
    display_df = display_df[['rank', 'model', 'win_rate', 'ci_range', 'elo_rating', 'wins', 'total_matches']]
    display_df.columns = ['Rank', 'Model', 'Win Rate (%)', '95% CI (±)', 'ELO Rating', 'Wins', 'Total Matches']
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Win Rate (%)": st.column_config.ProgressColumn(
                "Win Rate (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "95% CI (±)": st.column_config.TextColumn(
                "95% CI (±)",
                help="95% confidence interval for win rate (margin of error format)",
                width="medium"
            ),
            "ELO Rating": st.column_config.NumberColumn("ELO Rating", width="medium")
        }
    )



def render_task_detail_page(category):
    """Render detailed page for a specific task category."""
    use_mlflow = True
    df = get_cached_data(use_mlflow=use_mlflow)
    
    st.title(f"{category} Leaderboard")
    st.markdown(f"**Detailed performance analysis for {category} task**")
    
    # Back button
    if st.button("← Back to Overview"):
        st.session_state.selected_page = "Overview"
        st.rerun()
    
    # Category-specific leaderboard
    category_leaderboard = create_leaderboard(df, category)
    
    # Summary for this category
    category_matches = len(df[df['task_category'] == category])
    st.metric("Total Matches in Category", category_matches)
    
    st.markdown("---")
    
    # Detailed leaderboard
    st.subheader(f"{category} Rankings")
    
    # Check if we have any data for this category
    if not category_leaderboard:
        st.warning(f"No data available for {category} category.")
        st.info("Try checking a different category or verify your data source.")
        return
    
    display_df = pd.DataFrame(category_leaderboard)
    display_df = display_df[['rank', 'model', 'win_rate', 'ci_range', 'elo_rating', 'wins', 'total_matches']]
    display_df.columns = ['Rank', 'Model', 'Win Rate (%)', '95% CI (±)', 'ELO Rating', 'Wins', 'Total Matches']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Win Rate (%)": st.column_config.ProgressColumn(
                "Win Rate (%)",
                min_value=0,
                max_value=100,
                format="%.1f%%"
            ),
            "95% CI (±)": st.column_config.TextColumn(
                "95% CI (±)",
                help="95% confidence interval for win rate (margin of error format)",
                width="medium"
            ),
            "ELO Rating": st.column_config.NumberColumn("ELO Rating", width="medium")
        }
    )
    
    # Recent matches for this category
    st.markdown("---")
    st.subheader("Recent Matches")
    
    category_matches_df = df[df['task_category'] == category].copy()
    
    if category_matches_df.empty:
        st.info(f"No recent matches found for {category} category.")
    else:
        category_matches_df = category_matches_df.sort_values('timestamp', ascending=False).head(10)
        
        recent_display = category_matches_df[['timestamp', 'model_a', 'model_b', 'winner']].copy()
        recent_display['timestamp'] = recent_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        recent_display.columns = ['Timestamp', 'Model A', 'Model B', 'Winner']
        
        st.dataframe(recent_display, use_container_width=True, hide_index=True)

def main():
    """Main application logic."""
    # Enforce login before doing anything expensive or loading data
    require_company_login()
    
    # Initialize session state
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Overview"
    
    # Modern sidebar navigation
    st.sidebar.title("🏆 LLM Leaderboard")
    st.sidebar.markdown("---")
    
    # Always use MLflow data
    use_mlflow = True
    
    # Load data to get available categories
    df = get_cached_data(use_mlflow=use_mlflow)
    
    # Build navigation dynamically based on available data
    nav_items = [("Overview", "", "Main dashboard with all categories", False)]
    
    if not df.empty:
        available_categories = df['task_category'].unique().tolist()
        
        for category in available_categories:
            description = f"{category} analysis tasks"
            nav_items.append((category, "", description, False))
    
    st.sidebar.markdown("### Navigation")
    
    for page, emoji, description, is_coming_soon in nav_items:
        is_active = st.session_state.selected_page == page
        
        if is_coming_soon:
            # Disabled button for coming soon
            st.sidebar.button(
                f"{page} (Coming Soon)", 
                key=f"nav_{page}",
                help=description,
                use_container_width=True,
                disabled=True
            )
        else:
            if st.sidebar.button(
                page, 
                key=f"nav_{page}",
                help=description,
                use_container_width=True
            ):
                if st.session_state.selected_page != page:
                    st.session_state.selected_page = page
                    st.rerun()
    
    # Render appropriate page
    if st.session_state.selected_page == "Overview":
        render_overview_page()
    else:
        render_task_detail_page(st.session_state.selected_page)
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("This leaderboard tracks AI model performance across different task categories from our MLflow experiments.")
    st.sidebar.markdown("**Built with Streamlit** • **Powered by MLflow**")
    
    # Data refresh and update options
    st.sidebar.markdown("---")
    
    # Regular refresh button (always available)
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True, help="Clear cache and reload data"):
        st.cache_data.clear()
        st.rerun()
    
    # Manual data update button (only show if MLflow is available)
    source_type, _, _ = get_data_source_info()
    if source_type == "live":
        st.sidebar.markdown("### Update Saved Data")
        if st.sidebar.button("💾 Update Data File", use_container_width=True, help="Save current MLflow data to file for deployment"):
            try:
                from data_utils import update_saved_data
                with st.spinner("Updating saved data file..."):
                    success = update_saved_data()
                if success:
                    st.sidebar.success("✅ Data file updated!")
                    st.sidebar.info("💡 Commit and push to deploy this data.")
                else:
                    st.sidebar.error("❌ Failed to update data file.")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

if __name__ == "__main__":
    main() 