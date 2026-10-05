The task gives the axis calibration a person measured on this image, as in
WebPlotDigitizer: for each axis, two tick marks with their pixel position in
the image file and their value (`x_ticks`: `pixel_x` and `value`; `y_ticks`:
`pixel_y` and `value`), and whether each axis is linear or log (`x_scale`,
`y_scale`). Pixel positions are in the file's own pixel grid (`image_size` =
width, height): origin at the top-left corner, x to the right, y downward.
Use exactly this calibration to convert marker positions to values
(interpolate in log10 for a log axis) and report values in the space of the
given tick values. You do not need to read the tick labels.
