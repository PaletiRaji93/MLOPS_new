from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String


# ---------------------------------------------------------
# Project
# ---------------------------------------------------------
project_name = "dam_mcp_forecast"


# ---------------------------------------------------------
# Entity
# ---------------------------------------------------------
block = Entity(
    name="block",
    join_keys=["block_id"],
    description="15-minute DAM market time block",
)


# ---------------------------------------------------------
# Offline source
# ---------------------------------------------------------
dam_mcp_source = FileSource(
    name="dam_mcp_features_source",
    path="data/march_2025_features.parquet",
    timestamp_field="event_timestamp",
)


# ---------------------------------------------------------
# Feature View
# ---------------------------------------------------------
dam_mcp_features = FeatureView(
    name="dam_mcp_features",
    entities=[block],
    ttl=timedelta(days=365),
    schema=[
        Field(name="hour_sin", dtype=Float64),
        Field(name="hour_cos", dtype=Float64),
        Field(name="dow_sin", dtype=Float64),
        Field(name="dow_cos", dtype=Float64),
        Field(name="block_sin", dtype=Float64),
        Field(name="block_cos", dtype=Float64),

        Field(name="day_of_month", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
        Field(name="is_peak_hour", dtype=Int64),
        Field(name="is_morning_ramp", dtype=Int64),
        Field(name="is_off_peak", dtype=Int64),

        Field(name="dam_purchase_bid", dtype=Float64),
        Field(name="dam_sell_bid", dtype=Float64),
        Field(name="dam_mcv", dtype=Float64),
        Field(name="dam_volume", dtype=Float64),
        Field(name="dam_bid_imbalance", dtype=Float64),

        Field(name="rtm_purchase_bid", dtype=Float64),
        Field(name="rtm_sell_bid", dtype=Float64),
        Field(name="rtm_mcv", dtype=Float64),
        Field(name="rtm_volume", dtype=Float64),

        Field(name="mcp_spread", dtype=Float64),

        Field(name="dam_mcp_lag_1d", dtype=Float64),
        Field(name="dam_mcp_lag_2d", dtype=Float64),
        Field(name="dam_mcp_lag_7d", dtype=Float64),

        Field(name="dam_mcp_roll_4h", dtype=Float64),
        Field(name="dam_mcp_roll_24h", dtype=Float64),
        Field(name="dam_mcp_roll_std_24h", dtype=Float64),

        Field(name="avg_temp", dtype=Float64),
        Field(name="avg_humidity", dtype=Float64),
        Field(name="avg_windspeed", dtype=Float64),
        Field(name="avg_cloud_cover", dtype=Float64),
        Field(name="total_rainfall", dtype=Float64),

        Field(name="is_ipl_match", dtype=Int64),
        Field(name="is_event", dtype=Int64),
        Field(name="is_festival", dtype=Int64),
        Field(name="is_wedding_season", dtype=Int64),

        Field(name="impact", dtype=Float64),
    ],

    online=True,
    source=dam_mcp_source,
    tags={"team": "dam_mcp_forecasting"},
)


# ---------------------------------------------------------
# Feature Service
# ---------------------------------------------------------
dam_mcp_forecast_v1 = FeatureService(
    name="dam_mcp_forecast_v1",
    features=[dam_mcp_features],
)