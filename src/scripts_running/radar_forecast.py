from src.radar_forecast.read_and_forecast_radolan import make_predictions_rain_rasters_dir_compressed, \
    make_predictions_rain_rasters
from src.config_paths import dir_raster_radar_historical, path_csv_forecast_rain_radar
from src.config_variables import n_steps_predict

filenames_past, filenames_present, dfs_results_rois = make_predictions_rain_rasters_dir_compressed(
    dir_raster_radar_historical,
    n_steps_predict,
    make_predictions_rain_rasters,
    n_processes=1
)

# Remark: summation is performed without handling of normalization / units,
#   as this is not relevant for the model that is trained with the
#   historical rain forecasts
for name_roi, df in dfs_results_rois.items():
    dfs_results_rois.to_csv(path_csv_forecast_rain_radar)
