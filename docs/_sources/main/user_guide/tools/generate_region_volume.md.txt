# Region image generation

Generates an image of specified Allen institute brain regions (either solid,
or hollow).

```bash
cellfinder_gen_region_vol -s "Name of Structure 1" "Name of Structure 2" -a /path/to/atlas/annotations.nii -o output.nii
```

### Arguments
Run `cellfinder_gen_region_vol -h` to see all options.

* `-s` or `--structure-names` Structure names as a series of string 
(as per the reference atlas). If these are higher level structures 
(e.g. "Retrosplenial area"), it will include all subregions. If multiple 
structures are provided the final image will be the combination of all of them.
* `-a` or `--atlas` Path to the atlas annotations. The resulting image 
will be in the same coordinate space
* `-o` or `--output` Output name for the resulting `.nii` file

#### The following options may also need to be used:
* `--glass` Generate a hollow volume
* `--atlas-config` To supply your own, custom atlas configuration file