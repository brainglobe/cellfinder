# ROI transformation

Transforms ROI positions from ImageJ/FIJI into standard space.

When given an ImageJ ROI collection `.zip` files an a cellfinder registration 
output directory, an image of the ROIs in standard space will be generated.

```bash
cellfinder_roi_transform /path/to/rois.zip /path/to/registration/directory

```

### Arguments
Run `cellfinder_roi_transform -h` to see all options.

#### Positional
* ImageJ/FIJI ROI .zip collection.
* Cellfinder registration directory containing the downsampled image that 
the ROIs were drawn on

#### The following options may also need to be used:
* `-r` or `--reference` If the ROIs are not defined on the downsampled image 
in the `registration` directory (i.e. on the raw, non-downsampled data),
provide the path to this image here (e.g. the directory with a series 
of tiff files.)
* `-o` or `--output` Output filename. If not provided, will default to 
saving in the same directory as the ROIs
* `--registration-config` To supply your own, custom registration 
configuration file.
* `--selem` Size of the structuring element used to clean up the ROI image.
Default - 10. Increase if the ROI looks "fragmented", decrease if it looks too
"rounded".
* `--zfill` Increase this number to fill in gaps in your ROIs in Z (in case 
you drew your ROIs on downsampled data). Default: 2.
* `--debug` Debug mode. Save all intermediate files for diagnosis of 
software issues.