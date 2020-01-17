# Three dimensional viewer

**3D image viewing using [napari](http://napari.org/)**

**Work in progress, for the time being, I recommend 
[ClearVolume](https://imagej.net/ClearVolume) in 
[ImageJ](https://fiji.sc/#download)**

Unlike the [two dimensional cell viewer](./cell_view_2D.md), the data must fit
into RAM.
## Usage

``` bash
    cellfinder_view_3D -/path/to/image
```

### Arguments
#### Mandatory
* `-i` or `--img-paths` Can be the path of a nifti file, tiff file, 
tiff files folder or text file containing a list of paths
