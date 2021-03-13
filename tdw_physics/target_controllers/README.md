# tdw_physics target controllers

Controllers in this directory are for generating human benchmark stimuli and training data for models that will be compared to humans. The "target" refers to a single special object per scenario that will be the subject of a task prompt, e.g. "Will the red object hit the yellow area of the ground?"

Each controller will generate stimuli featuring a single type of physical scenario, detailed below. They can be called with 
```
python [controller_name.py] --arg1 value1 --arg2 value2 ...
```

Arguments common to all target_controllers are described below, followed by descriptions of each controller. Arguments specific to one controller are documented at the top of each script. 

Finally, we describe four use cases for using these controllers: (1) recreating stimuli from their script's arguments, (2) generating scenario-specific training data for models, (3) adding new types of variation to an existing scenario, and (4) designing a new scenario by subclassing off of an existing controller.

## Arguments

These arguments are common for every controller.

| Argument   | Type  | Default                                                      | Description                          |
| ---------- | ----- | ------------------------------------------------------------ | ------------------------------------ |
| `--dir`    | `str` | `"D:/" + dataset_dir` <br>`dataset_dir` is defined by each controller. | Root output directory.               |
| `--num`    | `int` | 3000                                                         | The number of trials in the dataset. |
| `--temp`   | `str` | D:/temp.hdf5                                                 | Temp path for incomplete files.      |
| `--width`  | `int` | 256                                                          | Screen width in pixels.              |
| `--height` | `int` | 256                                                          | Screen height in pixels.             |
| `--gpu`    | `int` | 0                                                            | Which gpu to run on  |
| `--random` | `int` | 1 | If 1, the random seed passed to the script will be ignored. Should be 0 for generating benchmark stimuli. |
| `--seed`   | `int` | 0 | The random seed for generating a batch of `--num` trials. If you want to regenerate stimuli, this needs to be set exactly as before. |
| `--run` | `int` | 1 | If 0, the controller will not be run, just initialized. |
| `--monochrome` | `int` | 0 | If 1, all the non-target and non-probe objects in a scene will have the same color (distinct from the target color.) | 
| `--room` | `str` | `"box"` | Which preset TDW room to use. `"tdw"` has tiles, more natural lighting, and windows, but runs more slowly. |
| `--save_passes` | `str` | `""` | Which image passes to save _as PNGs or MP4s_. A comma-separated list of items from `["_img", "_id", "_depth", "_normals", "_flow"]` | 
| `--save_movies` | `store_true` | `False` | Saved passes will be convered from PNGs to MP4s and the PNGs will be deleted after generation. |
| `--save_labels` | `store_true` | `False` | The script will create `metadata.json` and `trial_stats.json` files containing label information about each stimulus and the whole group, respectively. |

## Controllers

| Scenario        | Description                                                  | Script                                                       | Subclassed From                 |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------- |
| Drop  | A primitive Flex object is dropped onto a target object. | `drop.py` | `RigidBodiesDataset` |
| MultiDominoes | `M` dominoes are placed approximately in line with variable spacing. A probe domino is pushed into the first. | `dominoes.py` | `RigidBodiesDataset` |
| Tower | A tower is built out of primitive objects and optionally a "cap" target object is placed on top. A ball rolls (optionally down a ramp) into the tower. | `towers.py` | `MultiDominoes` |

## Use Cases

Here we briefly describe four ways to use these target_controllers. 

### 1. Regenerating stimuli from a file of script arguments

The stimulus files (`HDF5` and `MP4`) used for human benchmarking have high spatiotemporal resolution and are long. You may want to generate smaller versions of the exact same stimuli for testing models, or else regenerate the benchmark stimuli for some other reason.

Whenever a stimulus generation script completes, it produces two files, `commandline_args.txt` and `args.txt`, that contain all of the arguments passed to the script via the commandline and to the final controller (including default values), respectively. To regenerate a set of stimuli exactly, you would call 
```
python [controller_script.py] @[/YOUR/PATH/TO/commandline_args.txt] --dir [DIRECTORY_FOR_REGENERATED_STIMULI]
```
If you wanted to regenerate stimuli at a lower spatial resolution, you would simply add the arguments `--height [NEW_HEIGHT] --width [NEW_WIDTH]`. In general, arguments that follow the ones passed through `@commandline_args.txt` will overwrite the previous ones. 

Because `args.txt` includes arguments that have been "postprocessed" (see Designing a New Scenario below), calling `python controller.py @args.txt` is not supported. However, you can always inspect `args.txt` to see what value of some parameter was used to initialize the controller, including defaults not passed as commandline arguments.

Finally, if you try to regenerate stimuli and find that they don't match the originals, make sure that your `tdw_physics` repo has its HEAD at the same commit as the original stimuli; the commit can be found in the `metadata.json` file for the original stimuli.

### 2. Generating model training data from a scenario


