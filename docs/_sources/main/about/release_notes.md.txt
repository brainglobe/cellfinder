# Releases 

## Version 0.3.9 (2020-03-11)
### Added
Training can now be saved every N epochs using `--checkpoint-interval`
### Fixed
* Bug fixed in `cellfinder_xml_crop`
* Pandas dependencies updated thanks to [@amedyukhina](https://github.com/amedyukhina)

## Version 0.3.8 (2020-02-18)
### Fixed
* Fix memory leak in training

## Version 0.3.7 (2020-02-02)
### Changed
* Update curation interface

## Version 0.3.5 (2020-02-03)
### Main updates
* New all python training data generation interface, `cellfinder_curate`. 
Details [here](https://sainsburywellcomecentre.github.io/cellfinder/main/user_guide/training.html)


#### Developers
* Heatmap and converting cells to brainrender format moved to 
[neuro](https://github.com/SainsburyWellcomeCentre/neuro)
* Lots of utility functions moved to [imlib](https://github.com/adamltyson/imlib)
* Automatic deployment with Travis

## Version 0.3.2a (2020-01-10)
### Main updates
* Very basic Windows support (allthough some issues remain.)

* Pixel sizes are now defined in um (rather than mm). Users specifying a 
metadata file will not be affected. Those specifying the pixel sizes manually
will need to change their usage. Flags such as `--x-pixel-mm` are now 
`--x-pixel-um`.

* All cell detection parameters (e.g. `--soma-diameter`) are now defined in 
um or um<sup>3</sup> (rather than mm)

* Registration parameters are now defined via command-line arguments 
(e.g. `--freeform-use-n-steps`) rather than a config file.

* Installation is simplified (no need to install tensorflow separately)

#### Changed
* Updated default parameters to better work on noisy data.

#### Fixed
* Memory consumption during inference reduced

#### Developers
* Registration code is now in [amap](https://github.com/SainsburyWellcomeCentre/amap-python)
* Compatibility with tensorflow 2.1.0

## Version 0.3.1a (2019-10-31)
### Main updates
* Cube extraction is now carried out on the fly during classification (and 
not saved to disk). This prevents file system issues due to the generation of 
(potentially) hundreds of thousands of cells. 

#### Changed
* Likelihood during training of each individual augmentation transformation 
reduced to 10% (from 50%)

#### Fixed
* Bug fix in `cellfinder_view_cells` to deal with Napari api change

#### Developers
* New viewer to inspect batches of generated cubes 
(`cellfinder.classify.tools.batch_viewer`)
* Removed some old, unecessary tests
* Fixed a bug in `tests/tests/test_integration/test_detection.py`

## Version 0.3.0a (2019-10-29)
### Main updates
* Removed NiftyNet dependency. Classification is now implemented in `tf.keras`. 
**THIS MEANS YOU NEED TO TRAIN A NEW MODEL. OLD MODELS ARE INCOMPATIBLE!**

#### Added
* New `cellfinder_view_cells` command which uses [napari](http://napari.org/) 
view cells overlaid on raw data.
* New `cellfinder_gen_region_vol` command line tool to generate images of 
specific brain regions (for figures etc)
* `roi_transform` now supports ROIs generated in the original, full-size images
* New `cellfinder_cells_to_brainrender` to allow classified cells to be viewed 
in [BrainRender](https://github.com/BrancoLab/BrainRender)

#### Removed
`cellfinder_view_cells2D` function is removed, and replaced with 
`cellfinder_view_cells`.

#### Changed
* Cube scaling is now 3D (theoretically leading to better classification 
performance)
* `cellfinder_view_3D` now uses [napari](http://napari.org/).

#### Developers
* Brain image IO moved to [brainio](https://github.com/adamltyson/brainio)
* Reading a slurm environment (if present) moved to 
[slurmio](https://github.com/adamltyson/slurmio).
* Logging moved to [fancylog](https://github.com/adamltyson/fancylog)

## Version 0.2.12a (2019-09-24)
Added `roi_transform` command line tool to transform ImageJ ROI sets into
images in standard space.

## Version 0.2.11a (2019-09-23)
* Fixes a bug causing heatmaps to be scaled incorrectly

## Version 0.2.10a (2019-09-20)
Only minor changes to documentation and testing.

## Version 0.2.9a (2019-09-20)
#### Added
* New `--max-ram` to specify a maximum amount of RAM to use (in GB). Currently 
only affects processes that will scale to use as much RAM as is available 
(such as cube extraction).
* New `cellfinder_xml_scale` command line tool to rescale the cell positions 
within an XML file. For compatibility with other 
software, or if your data has been scaled after cell detection.

## Version 0.2.8a (2019-09-17)
**THIS VERSION HAS NOT BEEN FULLY TESTED**
#### Added
* Instructions for installing CUDA 9 and cuDNN via conda added.
* `cellfinder_region_summary` now has a `--sum-regions` flag combine child regions.
#### Fixed
* Bug which caused heatmaps to be calculated incorrectly is fixed.
* Bug which causes `--regions-list` input to `cellfinder_region_summary` to be
 parsed incorrectly is fixed.
* Bug which crashed summary csv generation if a cell is transformed incorrectly
 due to rounding errors is fixed.
 * Bug which crashed volume calculation in registration if a region is missing 
 is fixed.
 * Bug causing registration to crash if a custom registration file without a 
 `base_folder` entry is fixed.
 * `urllib3` version is specified to prevent atlas downloading from failing.
 
## Version 0.2.7a (2019-09-05)
#### Fixed
* A bug which caused the atlas not be downloaded when using 
`cellfinder_register` is fixed.
* Registration now respects the `--n-free-cpus` flag

## Version 0.2.6a (2019-09-05)
### Main updates
* Python 3.5 is no longer supported.
* Detected cell positions can now be transformed into standard space. By 
default, if cell classification and registration has been performed, an 
additional `cells_in_standard_space.xml` file will be generated. Use 
`--no-standard_space` to prevent this behaviour. This functionality can also 
be run using the standalone script `cellfinder_cell_standard`.

#### Added 
* Pixel sizes must still be specified, but instead of using the `-x`, `-y`, 
and `-z` flags, a `--metadata` flag can be used instead. Supported formats 
include [BakingTray](https://github.com/SainsburyWellcomeCentre/BakingTray) 
recipe files [mesoSPIM](https://github.com/mesoSPIM/mesoSPIM-control) 
 metadata files or 
 [cellfinder custom metadata files](https://github.com/SainsburyWellcomeCentre/cellfinder/tree/master/doc_build/examples/cellfinder_metadata.ini).
  If both pixel sizes and metadata are provided, the command line arguments 
  will take priority.

#### Fixed
* An installation bug (due to new requirements of 
[gputools](https://pypi.org/project/gputools/) is fixed.)

#### Deprecations
* The main command-line interface to cellfinder is now `cellfinder`, 
`cellfinder_run` has been removed (and may call an old version 
of the software). Please use `cellfinder` from now on.

## Version 0.2.5a (2019-07-31)
#### Fixed
* Bug in `cellfinder_gen_cubes` has been fixed 

#### Deprecations
* The main command-line interface to cellfinder will be `cellfinder`, 
`cellfinder_run` will stop working soon (and may call an old version 
of the software). Please use `cellfinder` from now on.

## Version 0.2.4a (2019-07-30)

**THIS VERSION HAS NOT BEEN FULLY TESTED**
**Use Version 0.2.3a if you don't need the new features**
### Main Updates
* Pixel sizes must now be specified with `-x`, `-y`, and `-z`.

#### Added 
* `cellfinder.IO.cells.get_cells()` can now read the `.yml` files output from 
the cell counting plugin of 
[MaSIV](https://github.com/SainsburyWellcomeCentre/masiv).
* Registration will now calculate a `volumes.csv` file which contains the 
volumes of each brain area in the atlas, in the sample brain.
* The `--summarise` flag will now save more information, including:
    * Cell counts combined across hemispheres
    * Cell densities in cells per mm<sup>3</sup>.
* Registration will also calculate the reverse transforms 
(i.e. sample -> atlas)
* A new `--figures` flag will create a subdirectory of 3D images in the 
coordinate space of the downsampled raw data (and registered atlas) that can 
be used to generate figures e.g. in [FIJI](https://fiji.sc). Currently three 
files will be generated by default.
    * **heatmap.nii** - This is a 3D histogram, showing cell densities 
    throughout the brain. Likely to be more useful than visualising cell
     positions for dense areas. By default, this is smoothed, and masked 
     (so that the smoothing doesn't "bleed" outside of the brain).
    * **glass_brain.nii** This is a binary image, just showing the brain 
    surface. Plotting this in 3D along with **heatmap.nii** gives a nice sense 
    of cell density throughout the brain. 
    [ClearVolume](https://imagej.net/ClearVolume) works well for this.
    * **boundaries.nii** Another binary image, but showing the divisions 
    between each region in the atlas. More useful for smaller, 3D figures to 
    delineate region boundaries. Also useful to assess registration by 
    overlaying on the raw, downsampled data. 


#### Fixed
* Bug on certain systems where cube extraction could use too much RAM, 
and crash the system has been fixed.

#### Changed
* Output directory structure has been reorganised, and simplified slightly.

#### Developers
* Code coverage increased to 58%.
* A lot of the code has been refactored, and (hopefully simplified). 
* Log files will now print more information (e.g. information about the git
repository, and internal parameters).


## Version 0.2.3a (2019-07-01)

### Main Updates
* Registration can now handle data in any orientation. The default is coronal, 
and in the NifTI standard orientation (origin is the most ventral, posterior, 
left voxel). If your data does not match this, use the `--orientation `, 
`--flip-x`, `--flip-y` and `--flip-z` tags.


#### Added 
* There is now a standalone registration command, `cellfinder_register`. This
will be the maintained python version of 
[aMAP](https://github.com/SainsburyWellcomeCentre/aMAP/wiki) going forward.
* There is now an option `--remove-intermediate` to remove all intermediate 
files generated by cellfinder. Use with caution.

#### Fixed
* All downsampled data, registered atlases and cell locations are in the same
coordinate space as the raw data (although possibly scaled). This fixes
issues #22 and #23, and allows overlay of the cell positions on the downsampled
data from the registration step.


#### Changed
* Cube extraction will now use as many CPU cores as available, slightly speeding
up this step. However, if you don't have much RAM (or a lot of CPU cores) 
see #36.
* Input data paths no longer need to end in the channel number. For channel 
referencing purposes, this will be set automatically, or can be overridden
with `--signal-channel-ids` and `--background-channel-id` (or `--channel-ids` 
for `cellfinder_gen_cubes`).
* By default, registration will now save downsampled versions of all channels. 
if you don't want these files, use `--no-save-downsampled`.
* Cube extraction will now default to extracting cubes of 50um x 50um x 100um. 
This can be overridden with `--x_pixel_mm_network` and `--y_pixel_mm_network`.
Currently there is no rescaling in z.
* Cube extraction now supports the output of 
[ROI sorter](https://github.com/SainsburyWellcomeCentre/Roi_sorter), rather 
than just xml.

#### Developers
* Code coverage increased to 32%.
* Logging now includes diagnostic information.

## Version 0.2.2a (2019-06-17)

### Main Updates
* Cellfinder can now be installed with a single command: 
```bash
    pip install git+https://github.com/adamltyson/cellfinder
```
* The atlas will now no longer be downloaded automatically upon installation 
of cellfinder, but on the first use of the atlas. To download in advance, 
`cellfinder_download` can be used.


#### Added 
* New function to load `.nrrd` files (in `cellfinder.tools.brain.io.load_any`).
* Data can now be passed as a directory of 2D .tif files (rather than just a 
text file of file paths).
* If a network was trained with a cubes of a different pixel size to that 
of the input data, then the pixel spacing of the output cubes can be changed
 (e.g. using: `--x-pixel-mm-network`).
* Cell (and candidate) positions can be saved as `.csv` (as well as `.xml`), 
by using `--save-csv`.
* `cellfinder_xml_crop` has been added. This curates an input `.xml` file and 
outputs a file with only those positions within given anatomical areas. 
Currently, the cells, and the atlas need to be in the same anatomical space.

#### Fixed
* Bug which caused cellfinder to hang when using part of a node managed by 
SLURM is [fixed](https://github.com/adamltyson/cellfinder/commit/ea6134ee6a8b811c1bbc81f5592dd8f3da06cd39).
* Bug [fixed](https://github.com/adamltyson/cellfinder/commit/cedf88472fa98443258e5439c5e3294edea6d3f2) 
which prevented cellfinder_run from progressing if a cube directory was 
present, but empty. Thanks to [ablot](https://github.com/ablot).
 * Bug [fixed](https://github.com/adamltyson/cellfinder/commit/f8561b486df04920db46909f616fe84e3a51aa16) 
 which prevented the `checkpoint` file from a previously trained network being
  copied to the output directory. Thanks to [ablot](https://github.com/ablot).

#### Changed
* If a cube cannot be extracted for any reason (such as proximity to the
edge of an image), the default is now to not save it (rather than save an 
empty cube in it's place). This can be overridden with `--save-empty-cubes`.
* Error messages when a cube cannot be extracted (due to proximity to the edge
of the image) have been clarified.


#### Developers
* Code is now automatically formatted with 
[Black](https://github.com/python/black), which requires Python >= 3.6 to run.
* Testing has been added (although currently there is only one test).
* All commits & pull requests will be build by [Travis](https://travis-ci.com).
* For development, cellfinder should be installed using `pip install -e .[dev]`
in the repository.


## Version 0.2.1a (2019-05-09)

### Main Updates
* Simple summary information can be generated with the `--summarise` flag. This will 
run registration (if it wasn't run allready) and associate each cell output from the cell 
detection step with a brain region. This info is saved as a csv file.
* The mouse brain atlas can be downloaded to any directory when installing cellfinder 
using the `--atlas-install-path` flag. This flag can also be used to point to an existing 
atlas installation (e.g. from [aMAP](https://github.com/SainsburyWellcomeCentre/amap_python)).
This will update the default config file so the correct atlas is sourced (and a second 
isn't downloaded.)
* Cellfinder *should* know what parts have allready been run, so if you re-run a 
`cellfinder_run` command, it won't repeat any parts of the pipeline.

#### Added 
* PDF documentation
* `cellfinder_gen_cubes` function to generate tiff cubes from an xml file 
independent of the rest of cellfinder. Useful for generating training data.
* `cellfinder_count_summary` function to combine a brain registered to the allen atlas 
with cell counts. Generates a csv file of cells per region.
* `cellfinder_region_summary` function to organise (align by brain area) csv files of
 summary cell counts from multiple brains.
* `cellfinder_view_cells2D` function to view cells overlaid on raw data. Uses hardly
any RAM, but I recommend [ROI sorter](https://github.com/SainsburyWellcomeCentre/Roi_sorter)
 instead.
 * `cellfinder_view_3D` function. A very rudimentary 3D viewer, but can load any file 
 type in use by cellfinder.
 * Some API docs

#### Fixed
* Logging that interfered with multiprocessing when the `--verbose` tag was used has 
been removed. This was an issue in case cellfinder was stopped in the middle of 
cell classification as it could cause GPU memory to not be released (and not show up 
in `nvidia-smi`).

#### Changed
* Documentation moved from pydocmd to sphinx.
* Installation requires configobj (as well as Cython).
* Install with `--dev` if you want to build the documentation yourself.


## Version 0.2.0a (2019-04-30)

### Main Updates
* Can run cell detection on N-channels
* Streamlined training of cell classification network via `cellfinder_train`
* Includes (rudimentary) integration of [aMAP](https://github.com/SainsburyWellcomeCentre/amap_python) for registration to the Allen brain atlas

#### Added 
* Further options to only run parts of the analysis
* Option `--tensorboard` to launch tensorboard automatically during training
* Entry point `cellfinder_view` to launch the (very simple) image/cell viewer
* Docs generated via pydocmd

#### Fixed
* Output of cell candidate detection step `cells.xml` can now be opened in [Roi Sorter](https://github.com/SainsburyWellcomeCentre/Roi_sorter)

#### Changed
* Docs generated by pydocmd means that installation info is currently [here](https://github.com/adamltyson/cellfinder/tree/master/docs/main/user_guide)


## Version 0.1.0a (2019-04-12) 

### Main Updates
First version compatible with NiftyNet 0.5.0, TensorFlow 1.12.0 and therefore CUDA 9/10.

#### Added 
* Option to specify external configuration file for cell candidate classification
* All results saved to a single directory
* Option to keep or discard bright spots classified as artifact during cell candidate detection
* Log file

#### Fixed
* Z positions of cell candidates when analysing a z-subset of the data fixed in the resulting .xml file
* Cell candidate detection .xml file only provides positive candidate types `<Type>1</Type>` for compatibility with [Roi Sorter](https://github.com/SainsburyWellcomeCentre/Roi_sorter)

#### Changed
* No longer relies on NiftyNet model zoo (although this may return)
* Much simplified installation (no manual cp/mv)
* Moving towards a single API, calls to n_count_python etc. broken
* Multiprocessing compatible with shared resources (e.g. HPC job schedulers)

#### Notes
* Developed with Python 3.5, but should work with 3.6/3.7. no Python 2 compatibility
* HPC compatibility only tested with SLURM, but should work with LSF & SGE


## Version 0.0.1a (2019-01-23) 

Final version compatible with niftynet 0.2.2 and tensorflow 1.4.1 (and therefore CUDA 8).
