"""
Spatial analysis of global weather patterns.

Creates:
  - Folium choropleth / marker maps
  - Plotly scatter geo maps
  - Regional aggregation and comparison
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FIGURES_DIR,
    LAT_COLUMN,
    LOCATION_COLUMN,
    LON_COLUMN,
    RANDOM_SEED,
    TARGET_VARIABLE,
)

logger = logging.getLogger(__name__)
np.random.seed(RANDOM_SEED)


def aggregate_by_location(df, columns=None):
    """Compute mean values per unique lat/lon location."""
    if columns is None:
        num_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        exclude = {LAT_COLUMN, LON_COLUMN, "year", "season_code"}
        columns = [c for c in num_cols if c not in exclude]

    agg_cols = [c for c in columns if c in df.columns]
    group_cols = [LOCATION_COLUMN, LAT_COLUMN, LON_COLUMN]
    group_cols = [c for c in group_cols if c in df.columns]

    location_df = df.groupby(group_cols)[agg_cols].mean().reset_index()
    logger.info("Aggregated %d locations", len(location_df))
    return location_df


def get_regional_stats(df, target_col=TARGET_VARIABLE):
    """Return per-country mean and std of the target variable."""
    if "country" not in df.columns:
        return pd.DataFrame()
    regional = (
        df.groupby("country")[target_col]
        .agg(mean_temp="mean", std_temp="std", count="count")
        .reset_index()
        .sort_values("mean_temp", ascending=False)
    )
    return regional


def create_temperature_map(location_df, target_col=TARGET_VARIABLE, save_path=None):
    """Build an interactive Folium map showing temperature at each location."""
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        logger.warning("folium not installed. Skipping interactive map.")
        return None

    if LAT_COLUMN not in location_df.columns or LON_COLUMN not in location_df.columns:
        logger.warning("Lat/Lon columns not found. Skipping map.")
        return None

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(m)

    q25 = location_df[target_col].quantile(0.25)
    q75 = location_df[target_col].quantile(0.75)

    for _, row in location_df.iterrows():
        temp = row.get(target_col, np.nan)
        if pd.isna(temp):
            continue
        color = "blue" if temp < q25 else ("orange" if temp < q75 else "red")
        loc_name = row.get(LOCATION_COLUMN, "Unknown")
        popup_text = f"{loc_name}<br>Avg Temp: {temp:.1f}°C"
        folium.CircleMarker(
            location=[row[LAT_COLUMN], row[LON_COLUMN]],
            radius=5, color=color, fill=True, fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200),
        ).add_to(cluster)

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "temperature_map.html"

    m.save(str(save_path))
    logger.info("Temperature map saved -> %s", save_path)
    return m


def create_plotly_scatter_geo(location_df, target_col=TARGET_VARIABLE, save_path=None):
    """Create a Plotly scatter geo map colored by temperature."""
    try:
        import plotly.express as px
    except ImportError:
        logger.warning("plotly not installed. Skipping plotly geo map.")
        return None

    if LAT_COLUMN not in location_df.columns or LON_COLUMN not in location_df.columns:
        return None

    df_plot = location_df.dropna(subset=[LAT_COLUMN, LON_COLUMN, target_col])
    fig = px.scatter_geo(
        df_plot, lat=LAT_COLUMN, lon=LON_COLUMN, color=target_col,
        hover_name=LOCATION_COLUMN if LOCATION_COLUMN in df_plot.columns else None,
        color_continuous_scale="RdBu_r",
        title="Global Mean Temperature by Location",
        labels={target_col: "Temperature (°C)"},
        projection="natural earth",
    )
    fig.update_geos(showland=True, landcolor="lightgray", showocean=True, oceancolor="lightblue")
    fig.update_layout(height=600, margin={"r": 0, "t": 50, "l": 0, "b": 0})

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "global_temperature_geo.html"

    fig.write_html(str(save_path))
    logger.info("Plotly geo map saved -> %s", save_path)
    return fig


def create_precipitation_map(location_df, save_path=None):
    """Create a scatter geo map for precipitation."""
    try:
        import plotly.express as px
    except ImportError:
        return None

    precip_col = "precip_mm"
    if precip_col not in location_df.columns:
        return None

    df_plot = location_df.dropna(subset=[LAT_COLUMN, LON_COLUMN, precip_col])
    fig = px.scatter_geo(
        df_plot, lat=LAT_COLUMN, lon=LON_COLUMN, color=precip_col, size=precip_col,
        hover_name=LOCATION_COLUMN if LOCATION_COLUMN in df_plot.columns else None,
        color_continuous_scale="Blues",
        title="Global Mean Precipitation by Location",
        labels={precip_col: "Precipitation (mm)"},
        projection="natural earth",
    )
    fig.update_geos(showland=True, landcolor="lightyellow")
    fig.update_layout(height=600)

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "global_precipitation_geo.html"

    fig.write_html(str(save_path))
    logger.info("Precipitation map saved -> %s", save_path)
    return fig


def create_geopandas_map(location_df, target_col=TARGET_VARIABLE, save_path=None):
    """Create a static choropleth-style scatter map using geopandas and matplotlib."""
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("geopandas not fully installed. Skipping static geo map.")
        return

    if LAT_COLUMN not in location_df.columns or LON_COLUMN not in location_df.columns:
        return

    world = None
    try:
        import geodatasets
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
    except Exception:
        pass
    if world is None:
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception:
            pass

    df_plot = location_df.dropna(subset=[LAT_COLUMN, LON_COLUMN, target_col])
    fig, ax = plt.subplots(figsize=(16, 8))

    if world is not None:
        world.plot(ax=ax, color="lightgray", edgecolor="white", linewidth=0.5)
    else:
        ax.set_facecolor("#d0e8f0")

    scatter = ax.scatter(
        df_plot[LON_COLUMN], df_plot[LAT_COLUMN],
        c=df_plot[target_col], cmap="RdBu_r", s=20, alpha=0.8, linewidths=0,
    )
    plt.colorbar(scatter, ax=ax, label="Mean Temperature (°C)", shrink=0.7)
    ax.set_title("Global Mean Temperature Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.tight_layout()

    if save_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_DIR / "global_temp_static_map.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Static geo map saved -> %s", save_path)


def run_spatial_analysis(df_clean):
    """Execute all spatial analyses."""
    logger.info("=== Spatial analysis started ===")

    location_df = aggregate_by_location(df_clean)
    regional_df = get_regional_stats(df_clean)

    from config import REPORTS_DIR
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    regional_df.to_csv(REPORTS_DIR / "regional_temperature_stats.csv", index=False)

    create_plotly_scatter_geo(location_df)
    create_precipitation_map(location_df)
    create_temperature_map(location_df)
    create_geopandas_map(location_df)

    logger.info("=== Spatial analysis complete ===")
    return location_df, regional_df
