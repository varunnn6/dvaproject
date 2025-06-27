import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Streamlit page configuration
st.set_page_config(page_title="Smartphone Sales Visualizer 📱", layout="wide")

# Color map for brands
color_map = {
    'Xiaomi': '#FF6F61',
    'Realme': '#6B5B95',
    'Huawei': '#88B04B',
    'Oppo': '#F7CAC9',
    'Samsung': '#92A8D1',
    'Sony': '#955251',
    'Apple': '#B565A7'
}

# Load CSV from GitHub or uploaded file
@st.cache_data
def load_data(file_url=None, uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
        else:
            df = pd.read_csv(file_url, encoding='utf-8', on_bad_lines='skip')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_url or uploaded_file, encoding='latin1', on_bad_lines='skip')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_url or uploaded_file, encoding='iso-8859-1', on_bad_lines='skip')
            except UnicodeDecodeError as e:
                st.error(f"Failed to decode file: {e}")
                return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

    # Validate required columns
    required_columns = {'Year', 'Brand', 'Model', 'Units Sold (Millions)'}
    if not required_columns.issubset(df.columns):
        st.error(f"Missing required columns. Found: {list(df.columns)}")
        return None

    # Clean data
    df['Year'] = df['Year'].astype(int)
    df['Model'] = df['Model'].fillna('Unknown')
    return df

# GitHub raw CSV URL (replace with your actual GitHub raw URL)
github_csv_url = "https://raw.githubusercontent.com/varunnn6/dvaproject/main/smartphone_sales2.csv"

# File uploader as fallback
uploaded_file = st.sidebar.file_uploader("Upload CSV file (optional)", type=["csv"])

# Load data
df = load_data(file_url=github_csv_url, uploaded_file=uploaded_file)
if df is None:
    st.stop()

# Sidebar for user inputs
st.sidebar.header("Smartphone Sales Visualizer 📱")
brands = sorted(df['Brand'].unique().tolist() + ['All'])
selected_brand = st.sidebar.selectbox("Select Brand", brands, index=brands.index('All'))

# Update model options based on selected brand
if selected_brand == 'All':
    models = ['None']
else:
    models = sorted(['All', 'None'] + df[df['Brand'] == selected_brand]['Model'].unique().tolist())
selected_model = st.sidebar.selectbox("Select Model", models, index=0)

# Year range selection
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

chart_types = ['Bar Chart', 'Pie Chart', 'Line Chart', 'Heatmap']
selected_chart = st.sidebar.selectbox("Select Chart Type", chart_types)

# Comparison charts
st.sidebar.subheader("Compare Charts")
compare_charts = st.sidebar.multiselect("Select Charts to Compare", chart_types)
compare_button = st.sidebar.button("Compare Charts")

# Show stats button
show_stats = st.sidebar.button("Show Stats")

# Function to render charts
def render_chart(chart_type, selected_brand, selected_model, year_range, fig=None, ax=None, is_comparison=False):
    if ax is None:
        ax = plt.gca()

    # Filter data by year range
    filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

    if chart_type == 'Heatmap':
        pivot = filtered_df.pivot_table(index='Brand', columns='Year', values='Units Sold (Millions)', aggfunc='sum', fill_value=0)
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap='coolwarm', ax=ax, cbar=True)
        ax.set_title("Heatmap: Smartphone Sales")
    else:
        if selected_model and selected_model not in ['None', 'All'] and selected_brand != 'All':
            filtered_df = filtered_df[(filtered_df['Brand'] == selected_brand) & (filtered_df['Model'] == selected_model)]
            if not filtered_df.empty:
                data = filtered_df['Units Sold (Millions)']
                years = filtered_df['Year']
                if chart_type == 'Bar Chart':
                    bars = ax.bar(years, data, color=color_map.get(selected_brand, 'gray'))
                    for bar in bars:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.1f}', 
                                ha='center', va='bottom', fontsize=8)
                    ax.set_title(f"{selected_model} Sales ({year_range[0]}-{year_range[1]})")
                elif chart_type == 'Line Chart':
                    ax.plot(years, data, marker='o', color=color_map.get(selected_brand, 'gray'))
                    for x, y in zip(years, data):
                        ax.text(x, y, f'{y:.1f}', ha='center', va='bottom', fontsize=8)
                    ax.set_title(f"{selected_model} Sales ({year_range[0]}-{year_range[1]})")
                elif chart_type == 'Pie Chart':
                    ax.text(0.5, 0.5, "Pie chart not applicable for single model", ha='center', va='center', fontsize=14)
                    ax.axis('off')
                ax.set_xlabel("Year")
                ax.set_ylabel("Units Sold (Millions)")
            else:
                ax.text(0.5, 0.5, "No data for selected model in selected year range", ha='center', va='center', fontsize=14)
                ax.axis('off')
        elif selected_brand == 'All':
            grouped = filtered_df.groupby(['Brand'])['Units Sold (Millions)'].sum().reindex(color_map.keys(), fill_value=0)
            if chart_type == 'Bar Chart':
                bars = grouped.plot(kind='bar', ax=ax, color=[color_map.get(b, 'gray') for b in grouped.index])
                for bar in ax.patches:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.1f}', 
                            ha='center', va='bottom', fontsize=8)
                ax.set_title(f"Total Smartphone Sales by Brand ({year_range[0]}-{year_range[1]})")
            elif chart_type == 'Pie Chart':
                grouped = grouped.sort_values(ascending=False)
                colors = [color_map.get(brand, 'gray') for brand in grouped.index]
                grouped.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors)
                ax.set_ylabel("")
                ax.set_title(f"Sales Distribution by Brand ({year_range[0]}-{year_range[1]})")
            elif chart_type == 'Line Chart':
                for brand in color_map.keys():
                    line_df = filtered_df[filtered_df['Brand'] == brand].groupby('Year')['Units Sold (Millions)'].sum()
                    ax.plot(line_df.index, line_df.values, label=brand, color=color_map.get(brand, 'gray'))
                ax.set_title(f"Total Sales Trends by Brand ({year_range[0]}-{year_range[1]})")
                ax.legend()
            ax.set_ylabel("Units Sold (Millions)")
        else:
            # Handle case where selected_model is 'All' or 'None'
            filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]
            grouped = filtered_df.groupby('Year')['Units Sold (Millions)'].sum()
            if chart_type == 'Bar Chart':
                bars = grouped.plot(kind='bar', ax=ax, color=color_map.get(selected_brand, 'gray'))
                for bar in ax.patches:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.1f}',
                            ha='center', va='bottom', fontsize=8)
                ax.set_title(f"{selected_brand} Total Sales ({year_range[0]}-{year_range[1]})")
            elif chart_type == 'Line Chart':
                grouped.plot(kind='line', ax=ax, marker='o', color=color_map.get(selected_brand, 'gray'))
                for x, y in zip(grouped.index, grouped.values):
                    ax.text(x, y, f'{y:.1f}', ha='center', va='bottom', fontsize=8)
                ax.set_title(f"{selected_brand} Total Sales ({year_range[0]}-{year_range[1]})")
            elif chart_type == 'Pie Chart':
                ax.text(0.5, 0.5, "Pie chart not applicable for single brand", ha='center', va='center', fontsize=14)
                ax.axis('off')
            ax.set_xlabel("Year")
            ax.set_ylabel("Units Sold (Millions)")
    return fig

# Main content
st.header("Smartphone Sales Visualizer 📱")

if not compare_button:
    # Display single chart
    with st.spinner("Rendering chart..."):
        fig, ax = plt.subplots(figsize=(10, 6))
        render_chart(selected_chart, selected_brand, selected_model, year_range, fig, ax)
        st.pyplot(fig)

else:
    # Display comparison charts
    if len(compare_charts) < 2:
        st.warning("Please select at least two charts to compare.")
    else:
        n_charts = len(compare_charts)
        cols = 2 if n_charts <= 2 else 3
        rows = (n_charts + cols - 1) // cols
        with st.spinner("Rendering comparison charts..."):
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
            axes = np.array(axes).flatten() if n_charts > 1 else [axes]
            for idx, chart_type in enumerate(compare_charts):
                render_chart(chart_type, selected_brand, selected_model, year_range, fig, axes[idx], is_comparison=True)
            for idx in range(len(compare_charts), len(axes)):
                axes[idx].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

# Show statistics
if show_stats:
    if selected_brand == 'All':
        st.warning("Select a single brand to see statistics.")
    else:
        filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        filtered_df = filtered_df[filtered_df['Brand'] == selected_brand]
        if selected_model not in ['All', 'None']:
            filtered_df = filtered_df[filtered_df['Model'] == selected_model]
        model_text = 'All Models' if selected_model in ['All', 'None'] else selected_model
        brand_data = filtered_df['Units Sold (Millions)']
        if brand_data.empty:
            st.warning(f"No data available for {selected_brand} ({model_text}) in the selected year range.")
        else:
            stats_text = f"""
            📊 **{selected_brand} ({model_text}) Statistics ({year_range[0]}-{year_range[1]})**:
            - Total Sales: {brand_data.sum():.2f} Million Units
            - Average Sales: {brand_data.mean():.2f} Million Units
            - Max Sales: {brand_data.max():.2f} Million Units
            - Min Sales: {brand_data.min():.2f} Million Units
            - Std Deviation: {brand_data.std():.2f}
            """
            st.markdown(stats_text)
