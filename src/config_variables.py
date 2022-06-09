from datetime import datetime
import pytz

name_column_rain_history = 'rain_rate [mm/min]'
name_column_inflow = 'wtp_inflow [l/s]'

variables_external = [
    'Level_1 [mNHN]',
    'Level_2 [mNHN]',
    'Level_3 [mNHN]',
    'level_storage_sewer [mNHN]',
    name_column_rain_history
]

# columns to plot for level and sku
column_level_plot = 'Level_1' #TODO
column_sku_plot = 'level_storage_sewer [mNHN]'

date_start_train = datetime(2017, 1, 1, 0, 0, tzinfo=pytz.utc)
date_end_train = datetime(2018, 12, 31, 23, 45, tzinfo=pytz.utc)  # end without rain
date_start_test = datetime(2019, 1, 1, 0, 0, tzinfo=pytz.utc)
date_end_test = datetime(2019, 12, 31, 23, 45, tzinfo=pytz.utc)  # end without rain

# TODO: only for testing
date_start_train = datetime(2018, 6, 1, 0, 0, tzinfo=pytz.utc)
date_end_train = datetime(2018, 12, 31, 23, 45, tzinfo=pytz.utc)  # end without rain
date_start_test = datetime(2019, 1, 1, 0, 0, tzinfo=pytz.utc)
date_end_test = datetime(2019, 6, 29, 23, 45, tzinfo=pytz.utc)  # end without rain

n_steps_predict = 2  # tODO: 10

min_lag = -6