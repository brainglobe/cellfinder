# Transforming cells into standard space

To transform cell positions from the coordinate space of the raw data to that 
of another coordinate framework (usually the atlas).

``` bash
cellfinder_cell_standard --cells /path/to/classified_cells.xml --transformation /path/to/control_point_file.nii --ref /path/to/reference_image.nii -o /path/to/output/directory -x 0.002 -y 0.002 -z 0.005
```


### Arguments
#### Mandatory
* `--cells`  Path of the xml file containing cells to be transformed.                        
* `--transformation` Path to the control point file 
(from niftyreg or cellfinder registration)
* `--reference` Reference nii image, in the same space as the downsampled raw 
data
 * `-o` or `--output-dir` Output directory
 
**Either**
* `-x` or `--x-pixel-mm` Pixel spacing of the data in the first dimension, 
specified in mm.
* `-y` or `--y-pixel-mm` Pixel spacing of the data in the second dimension, 
specified in mm.
* `-z` or `--z-pixel-mm` Pixel spacing of the data in the third dimension, 
specified in mm.

**Or**
* `--metadata` Metadata file containing pixel sizes (supported formats 
include [BakingTray](https://github.com/SainsburyWellcomeCentre/BakingTray) 
recipe files [mesoSPIM](https://github.com/mesoSPIM/mesoSPIM-control) 
 metadata files or 
 [cellfinder custom metadata files](https://github.com/SainsburyWellcomeCentre/cellfinder/tree/master/doc_build/examples/cellfinder_metadata.ini).
  If both pixel sizes and metadata are provided, the command line arguments 
  will take priority.


#### The following options can also be used:
* `--transform-all` Transform all cell positions (including artifacts).
* `----registration-config` To supply your own, custom registration 
configuration file.


**Performance/debugging**

* `-V` or `--verbose` Increase verbosity of statements printed to console (all
debug information is saved to file regardless of this flag)
* `--n-free-cpus` The number of CPU cores on the machine to leave 
unused by the program to spare resources.


**All other options (and their defaults) can be round by running:** 

`cellfinder_cell_standard -h`
