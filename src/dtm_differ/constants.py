"""Constants for nodata values and other configuration."""
   
# Standard nodata value for floating-point rasters (meters)
NODATA_FLOAT = -9999.0
   
# Nodata value for direction raster (int8 range: -128 to 127)
# Using 127 as it's outside valid range (-1, 0, 1)
NODATA_DIRECTION = 127.0
   
# Nodata value for ranked movement (0 = unclassified, so 0 is valid)
# Consider using -1 or None for actual nodata
NODATA_RANK = 0.0