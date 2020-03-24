# Usage

``` bash
    cellfinder -s signal_channel_images  optional_signal_channel_images -b background_channel_images -o /path/to/output_directory -x 2 -y 2 -z 5
```

### Arguments
#### Mandatory
* `-s` or `--signal-planes-paths` Path to the directory of the signal files. 
Can also be a text file pointing to the files. **There can be as many signal 
channels as you like, and each will be treated independently**. 
* `b` or `--signal-planes-path` Path to the directory of the background files. 
Can also be a text file pointing to the files.  **This background channel will
 be used for all signal channels**.
 * `-o` or `--output-dir` Output directory for all intermediate and final 
results

**Either**
* `-x` or `--x-pixel-um` Pixel spacing of the data in the first dimension, 
specified in um.
* `-y` or `--y-pixel-um` Pixel spacing of the data in the second dimension, 
specified in um.
* `-z` or `--z-pixel-um` Pixel spacing of the data in the third dimension, 
specified in um.

**Or**
* `--metadata` Metadata file containing pixel sizes (any format supported 
by [micrometa](https://github.com/adamltyson/micrometa) can be used).
  If both pixel sizes and metadata are provided, the command line arguments 
  will take priority.


#### The following options can also be used:

**Additional options**
* `--register` Register the background channel to the Allen brain atlas
* `--summarise` Generate summary csv files showing how many cells are in 
each brain area (will also run registration if not specified.
* `--signal-channel-ids` Channel ID numbers, in the same order as 
* `--figures` Generate figures


**Only run parts of cellfinder**

If for some reason you don't want some parts of cellfinder to run, you can use 
the following options. If a part of the pipeline is required by another 
part it will be run (i.e. `--no-detection` won't do anything unless 
`--no-classification` is also used). Cellfinder will attempt to work out 
what parts of the pipeline have allready been run (in a given output 
directory) and not run them again if appropriate.


* `--no-detection` Don't run cell candidate detection
* `--no-classification` Don't run cell classification
* `--no-standard_space` Dont convert cell positions to standard space. 
Otherwise will run automatically if registration and classification has run.


**Figures options**

Figures cannot yet be customised much, but the current options are here:

* `--no-heatmap` Don't generate a heatmap of cell locations
* `--heatmap-bin` Heatmap bin size (um of each edge of histogram cube)
* `--heatmap-smoothing` Gaussian smoothing sigma, in um.
* `--no-mask-figs` Don't mask the figures (removing any areas outside the 
brain, from e.g. smoothing)


**Performance/debugging**
* `--debug` Increase verbosity of statements printed to console and save all 
intermediate files.
* `--n-free-cpus` The number of CPU cores on the machine to leave 
unused by the program to spare resources.
* `--max-ram` Maximum amount of RAM to use (in GB) - **not currently fully 
implemented for all parts of cellfinder**

Useful for testing or if you know your cells are only in a specific region
* `--start-plane` The first plane to process in the Z dimension
* `--end-plane` The last plane to process in the Z dimension

**Cell candidate detection**

* `--save-planes` Whether to save the individual planes after 
processing and thresholding. Useful for debugging.
* `--outlier-keep` Dont remove putative cells that fall outside initial
clusters
* `--artifact-keep` Save artifacts into the initial xml file
* `--max-cluster-size` Largest putative cell cluster (in cubic um) where 
splitting should be attempted.  **Default: 100000**
* `--soma-diameter` The expected soma size in um in the x/y dimensions. 
 **Default: 16**
* `--ball-xy-size` The size in um of the ball used
for the morphological filter in the x/y dimensions. **Default: 6**
* `--ball-z-size` The size in um of the ball used 
for the morphological filter in the z dimension.  **Default: 15**
* `--ball-overlap-fraction` The fraction of the ball that has to cover 
thresholded pixels for the centre pixel to be considered a nucleus pixel. 
 **Default: 0.6**
* `--log-sigma-size` The filter size used in the Laplacian of Gaussian filter 
to enhance the cell intensities. Given as a fraction of the soma-diameter.
**Default: 0.2**
* `--threshold` The cell threshold, in multiples of the standard deviation 
above the mean. **Default: 10**
* `--soma-spread-factor` Soma size spread factor (for splitting up 
cell clusters). **Default: 1.4**

**Cell candidate classification**

* `--trained-model` To use your own network (not the one 
supplied with cellfinder) specify the model file.
* `--model-weights` To use pretrained model weights. Ensure that this model 
matches the `--network-depth` parameter.
* `--network-depth`. Resnet depth (based on 
[He et al. (2015)](https://arxiv.org/abs/1512.03385) **Default: 50**
* `--batch-size` Batch size for classification. Can be adjusted depending on 
GPU memory. **Default: 32**

*You shouldn't need to change these:*
* `--x-pixel-um-network` The pixel size (in microns, in the first dimension) 
that the machine learning network was trained on.  Set this to adjust the 
pixel sizes of the extracted cubes. **Default 1**
* `--y-pixel-um-network` The pixel size (in microns, in the second dimension) 
that the machine learning network was trained on.  Set this to adjust the 
pixel sizes of the extracted cubes. **Default 1**
* `--z-pixel-um-network` The pixel size (in microns, in the third dimension) 
that the machine learning network was trained on.  Set this to adjust the 
pixel sizes of the extracted cubes. **Default 5**
* `--cube-width` The width of the cubes to extract in pixels (must be even). 
**Default 50**
* `--cube-height` The height of the cubes to extract in pixels (must be even). 
**Default 50**
* `--cube-depth` The depth (z)) of the cubes to extract in pixels
 (must be even). **Default 20**
* `--save-empty-cubes` If a cube cannot be extracted (e.g. to close to the 
edge of the image), save an empty cube instead. Useful to keep track of all 
cell candidates.


**Registration to atlas**
 * `--registration-config` To supply your own, custom registration
  configuration file.
  
 If the supplied data does not match the NifTI standard 
 orientation (origin is the most ventral, posterior, left voxel), 
 then the atlas can be flipped to match the input data:
 * `--flip-x` Flip the sample brain along the first dimension for 
 atlas registration
 * `--flip-y` Flip the sample brain along the second dimension for 
 atlas registration
 * `--flip-z` Flip the sample brain along the third dimension for 
 atlas registration
  * `--orientation` The orientation of the sample brain `coronal`, `saggital`
 or `horizontal`
 


**Standard space options**
* `--transform-all` Transform all cell positions (including artifacts).

**Atlas specification**

When cellfinder runs, the appropriate machine learning models and 
reference atlas will be downloaded (if not previously done so). If you would 
like to download in advance to save time (or if you will not have an internet
connection) please use `cellfinder_download`.

Currently, the only supported atlas is the Allen reference mouse 
brain, originally from [here](http://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas).
The atlas will download into `~/.cellfinder/atlas`. 

In the future, other atlases will be available, and you will be able to choose
which is downloaded.

If you want to modify the cellfinder download, use:
* `--atlas-install-path` Supply a path to download the atlas elsewhere. This 
should also update the default `registration.conf.custom` file so that the correct 
atlas is sourced. Alternatively, use this command to tell cellfinder where an 
existing atlas is, to save it being downloaded twice. (Requires 20GB 
disk space)
* `--atlas download path` The path to download the atlas into. 
(Requires 1.2GB disk space). Defaults to `/tmp`.

**Historical options**

*If you are a new cellfinder user, it is unlikely you want to use these 
options. They are intended for those migrating from earlier versions of the 
software*

If you have channel numbers that you'd like to carry over into cellfinder
* `--signal-channel-ids`. Channel ID numbers, in the same order as 'signal-planes-paths.
Will default to '0, 1, 2' etc, but maybe useful to specify.
* `--background-channel-id` Channel ID number, corresponding to 
'background-planes-path'

------------------------------------------------------------------

#### Further help
All cellfinder options can be found by using the `-h` flag

```bash
cellfinder -h
```

If you have any issues, common issues can be found 
[here](troubleshooting.md). If you're still having issues, 
[get in touch](mailto:adam.tyson@ucl.ac.uk?subject=Cellfinder%20troubleshooting),
or even better [raise an issue on Github](https://github.com/adamltyson/cellfinder/issues/new).