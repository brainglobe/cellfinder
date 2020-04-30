# Converting cells to brainrender format

**Temporary tool until a Cellfinder -> BrainRender bridge is finalised**

To convert cell positions (e.g. in `cells_in_standard_space.xml`) to 
a format that can be used in 
[BrainRender](https://github.com/BrancoLab/BrainRender).


```bash
    cellfinder_cells_to_brainrender cells_in_standard_space.xml exported_cells.h5
```

### Arguments
Run `cellfinder_cells_to_brainrender -h` to see all options.

#### Positional
* Cellfinder cells file to be converted
* Output filename. Should end with '.h5'


#### The following options may also need to be used:
* `-x` or `--x-pixel-size` Pixel spacing of the data that the cells are 
defined in, in the first dimension, specified in um. (Default: 10)
* `-y` or `--y-pixel-size` Pixel spacing of the data that the cells are 
defined in, in the second dimension, specified in um. (Default: 10)
* `-z` or `--z-pixel-size` Pixel spacing of the data that the cells are 
defined in, in the third dimension, specified in um. (Default: 10)
* `--max-z` Maximum z extent of the atlas, specified in um. (Default: 13200)
* `--hdf-key` HDF identifying key. If this has changed, it must be specified
in the call to `BrainRender.scene.Scene.add_cells_from_file()`


### To visualise this file in BrainRender
```python
from BrainRender.scene import Scene
scene = Scene(jupyter=True)
scene.add_cells_from_file("exported_cells.h5")
scene.render()
```