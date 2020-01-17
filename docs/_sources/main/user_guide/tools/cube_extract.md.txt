# Cube extraction

To extract tiff cubes for 
cell classification or training. Useful so you don't need to save thousands of
cubes for future use.

``` bash
    cellfinder_gen_cubes --cells /path/to/cells.xml -i /path/to/image/channel.txt /path/to/any/other/image/channels.txt -o /path/to/output_directory -x x_pixel_in_mm -y y_pixel_in_mm -z z_pixel_in_mm
```

### Arguments
#### Mandatory
* `--cells`  Path of the xml file (or Roi sorter output directory)containing the ROIs to be extracted.                        
* `-i` or `--img-paths` Path to the directory of the image files. 
Can also be a text file pointing to the files. **There can be as many 
channels as you like**.
 * `-o` or `--output-dir` Directory to save the cubes into
 
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
  
 **Cube extraction generates a large number (100,000+) of small files. Reading and writing these on shared (e.g. network) storage 
can cause issues for other users. It is strongly recommended to set 
`--output-dir` as local (preferably fast) storage.**

#### The following options can also be used:
**Cube definitions**

* `--cube-width` The width of the cubes to extract (must be even) [default 50]
* `--cube-height` The height of the cubes to extract (must be even) [default 50]
* `--cube-depth` The depth of the cubes to extract  [default 20]


**Performance/debugging**

* `--debug` Debug mode. Will increase verbosity of logging and save all 
intermediate files for diagnosis of software issues.
* `--n-free-cpus` The number of CPU cores on the machine to leave 
unused by the program to spare resources.
