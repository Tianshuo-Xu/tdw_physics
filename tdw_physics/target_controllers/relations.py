from argparse import ArgumentParser
import h5py, json, copy, importlib
import numpy as np
from enum import Enum
import random
import stopit
from typing import List, Dict, Tuple
from tdw.tdw_utils import TDWUtils
from tdw.librarian import ModelRecord, MaterialLibrarian
from tdw_physics.rigidbodies_dataset import (RigidbodiesDataset,
                                             get_random_xyz_transform,
                                             get_range,
                                             handle_random_transform_args)
from tdw_physics.util import (MODEL_LIBRARIES,
                              get_parser,
                              xyz_to_arr,
                              arr_to_xyz,
                              str_to_xyz)
from tdw_physics.target_controllers.dominoes import (get_args,
                                                     none_or_str,
                                                     none_or_int)
from tdw_physics.target_controllers.playroom import Playroom

MODEL_NAMES = [r.name for r in MODEL_LIBRARIES['models_full.json'].records if not r.do_not_use]
PRIMITIVE_NAMES = [r.name for r in MODEL_LIBRARIES['models_flex.json'].records if not r.do_not_use]
SPECIAL_NAMES =[r.name for r in MODEL_LIBRARIES['models_special.json'].records if not r.do_not_use]
ALL_NAMES = MODEL_NAMES + SPECIAL_NAMES + PRIMITIVE_NAMES

M = MaterialLibrarian()
MATERIAL_TYPES = M.get_material_types()
MATERIAL_NAMES = {mtype: [m.name for m in M.get_all_materials_of_type(mtype)] \
                  for mtype in MATERIAL_TYPES}

## Relation types
class Relation(Enum):
    contain = 'contain'
    support = 'support'
    occlude = 'occlude'
    null = 'null'

    def __str__(self):
        return self.value
    

def get_relational_args(dataset_dir: str, parse=True):

    common = get_parser(dataset_dir, get_help=False)
    domino, domino_postproc = get_args(dataset_dir, parse=False)
    parser = ArgumentParser(parents=[common, domino], conflict_handler='resolve',
                            fromfile_prefix_chars='@')

    ## Relation type
    parser.add_argument("--relation",
                        type=Relation,
                        choices=list(Relation),
                        help="Which relation type to construct")


    ## Object types
    parser.add_argument("--container",
                        type=none_or_str,
                        default="b04_bowl_smooth",
                        help="comma-separated list of container names")
    parser.add_argument("--target",
                        type=none_or_str,
                        default="b04_clownfish",
                        help="comma-separated list of target object names")
    parser.add_argument("--distractor",
                        type=none_or_str,
                        default="b05_lobster",
                        help="comma-separated list of distractor object names")

    ## Object scales
    parser.add_argument("--cscale",
                        type=str,
                        default="[1.0,2.0]",
                        help="scale of container")
    parser.add_argument("--tscale",
                        type=str,
                        default="[1.0,2.0]",
                        help="scale of target object")
    parser.add_argument("--dscale",
                        type=str,
                        default="[1.0,2.0]",
                        help="scale of distractor")

    ## Changed defaults
    parser.add_argument("--room",
                        type=str,
                        default="tdw",
                        help="Which room to be in")
    parser.add_argument("--zscale",
                        type=str,
                        default="-1.0",
                        help="scale of target zone")
    parser.add_argument("--zcolor",
                        type=none_or_str,
                        default=None,
                        help="comma-separated R,G,B values for the target zone color. None is random")
    parser.add_argument("--zmaterial",
                        type=none_or_str,
                        default=None,
                        help="Material name for zone. If None, samples from material_type")
    parser.add_argument("--material_types",
                        type=none_or_str,
                        default="Wood,Metal,Ceramic",
                        help="Which class of materials to sample material names from")    

    def postprocess(args):

        args.container = [nm for nm in args.container.split(',') if nm in ALL_NAMES]
        args.target = [nm for nm in args.target.split(',') if nm in ALL_NAMES]
        args.distractor = [nm for nm in args.distractor.split(',') if nm in ALL_NAMES]

        args.zscale = handle_random_transform_args(args.zscale)
        args.cscale = handle_random_transform_args(args.cscale)
        args.tscale = handle_random_transform_args(args.tscale)
        args.dscale = handle_random_transform_args(args.dscale)

        return args

    args = parser.parse_args()
    args = postprocess(args)

    return args

class RelationArrangement(Playroom):

    def __init__(self, port=1071,
                 container=PRIMITIVE_NAMES,
                 target=PRIMITIVE_NAMES,
                 distractor=PRIMITIVE_NAMES,
                 container_scale_range=[1.0,1.0],
                 target_scale_range=[1.0,1.0],
                 distractor_scale_range=[1.0,1.0],                 
                 **kwargs):

        super().__init__(port=port, **kwargs)

        ## object types
        self._container_types = self.set_types(container)
        self._target_types = self.set_types(target)
        self._distractor_types = self.set_types(distractor)

        ## object scales
        self.container_scale_range = container_scale_range
        self.target_scale_range = target_scale_range
        self.distractor_scale_range = distractor_scale_range
        

        print("sampling containers from", [(r.name, r.wcategory) for r in self._container_types], len(self._container_types))
        print("sampling targets from", [(r.name, r.wcategory) for r in self._target_types], len(self._target_types))
        print("sampling distractors from", [(r.name, r.wcategory) for r in self._distractor_types], len(self._distractor_types))

    def is_done(self, resp: List[bytes], frame: int) -> bool:
        return frame > 60

    def set_types(self, olist):
        tlist = self.get_types(olist,
                               libraries=["models_full.json", "models_special.json", "models_flex.json"],
                               categories=None,
                               flex_only=False,
                               size_min=None, size_max=None)
        return tlist

    def _write_static_data(self, static_group: h5py.Group) -> None:
        # randomization
        try:
            static_group.create_dataset("room", data=self.room)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("seed", data=self.seed)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("randomize", data=self.randomize)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("trial_seed", data=self.trial_seed)
        except (AttributeError,TypeError):
            pass
        try:
            static_group.create_dataset("trial_num", data=self._trial_num)
        except (AttributeError,TypeError):
            pass

    def _place_camera(self) -> List[dict]:
        commands = []
        a_pos = self.get_random_avatar_position(radius_min=self.camera_radius_range[0],
                                                radius_max=self.camera_radius_range[1],
                                                angle_min=self.camera_min_angle,
                                                angle_max=self.camera_max_angle,
                                                y_min=self.camera_min_height,
                                                y_max=self.camera_max_height,
                                                center=TDWUtils.VECTOR3_ZERO,
                                                reflections=self.camera_left_right_reflections)
        self._set_avatar_attributes(a_pos)
        commands.extend([
            {"$type": "teleport_avatar_to",
             "position": self.camera_position},
            {"$type": "look_at_position",
             "position": self.camera_aim},
            {"$type": "set_focus_distance",
             "focus_distance": TDWUtils.get_distance(a_pos, self.camera_aim)}
        ])

        return commands

    def get_trial_initialization_commands(self) -> List[dict]:
        commands = []
    
        ## randomization across trials
        if not(self.randomize):
            self.trial_seed = (self.MAX_TRIALS * self.seed) + self._trial_num
            random.seed(self.trial_seed)
        else:
            self.trial_seed = -1 # not used

        ## place "zone" (i.e. a mat on the floor)
        commands.extend(self._place_target_zone())

        ## place container

        ## place target
        commands.extend(self._place_target_object())

        ## teleport the avatar
        commands.extend(self._place_camera())

        ## place distractor

        return commands
        

if __name__ == '__main__':

    import platform, os
    args = get_relational_args("relational")
    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":0." + str(args.gpu)
        else:
            os.environ["DISPLAY"] = ":0"

        launch_build = True
    else:
        launch_build = True

    print(args.relation)

    RC = RelationArrangement(
        ## objects
        container=args.container,
        target=args.target,
        distractor=args.distractor,

        ## scales
        zone_scale_range=args.zscale,
        container_scale_range=args.cscale,
        target_scale_range=args.tscale,
        distractor_scale_range=args.dscale,

        ## common
        launch_build=launch_build,
        port=args.port,
        room=args.room,
        randomize=args.random,
        seed=args.seed,
        flex_only=False
    )

    if bool(args.run):
        RC.run(num=args.num,
                 output_dir=args.dir,
                 temp_path=args.temp,
                 width=args.width,
                 height=args.height,
                 framerate=args.framerate,
                 save_passes=args.save_passes.split(','),
                 save_movies=args.save_movies,
                 save_labels=args.save_labels,
                 save_meshes=args.save_meshes,
                 write_passes=args.write_passes,
                 args_dict=vars(args)
        )
    else:
        RC.communicate({"$type": "terminate"})
    

    


    
