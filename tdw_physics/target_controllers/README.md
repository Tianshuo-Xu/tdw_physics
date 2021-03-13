# tdw_physics target controllers

Controllers in this directory are for generating human benchmark stimuli and training data for models that will be compared to humans. The "target" refers to a single special object per scenario that will be the subject of a task prompt, e.g. "Will the red object hit the yellow area of the ground?"

Each controller will generate stimuli featuring a single type of physical scenario, detailed below. They can be called with 
```
python [controller_name.py] --arg1 value1 --arg2 value2 ...
```

Arguments common to all target_controllers are described below, followed by descriptions of each controller. Arguments specific to one controller are documented at the top of each script. 

Finally, we describe three use cases for using these controllers: (1) recreating stimuli from their script's arguments, (2) generating scenario-specific training data for models, and (3) designing a new scenario by subclassing off of an existing controller.

## Arguments

These arguments are common for every controller.

| Argument   | Type  | Default                                                      | Description                          |
| ---------- | ----- | ------------------------------------------------------------ | ------------------------------------ |
| `--dir`    | `str` | `"D:/" + dataset_dir` <br>`dataset_dir` is defined by each controller. | Root output directory.               |
| `--num`    | `int` | 3000                                                         | The number of trials in the dataset. |
| `--temp`   | `str` | D:/temp.hdf5                                                 | Temp path for incomplete files.      |
| `--width`  | `int` | 256                                                          | Screen width in pixels.              |
| `--height` | `int` | 256                                                          | Screen height in pixels.             |

## Controllers

| Scenario        | Description                                                  | Script                                                       | Type                 |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------- |
| Drop  | A primitive Flex object is dropped onto a target object. | `drop.py` | `RigidBodiesDataset` |
