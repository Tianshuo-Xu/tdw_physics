import sys, os
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from tdw.librarian import ModelRecord, MaterialLibrarian
from tdw_physics.target_controllers.dominoes import Dominoes, get_args
from tdw_physics.flex_dataset import FlexDataset

class FlexDominoes(Dominoes, FlexDataset):

    def __init__(self, port: int = 1071, **kwargs):

        Dominoes.__init__(self, port=port, **kwargs)
        self._clear_flex_data()

    def get_trial_initialization_commands(self) -> List[dict]:

        # clear the flex data
        FlexDataset.get_trial_initialization_commands(self)
        return Dominoes.get_trial_initialization_commands(self)

    def _get_send_data_commands(self) -> List[dict]:
        return FlexDataset._get_send_data_commands(self)

    def add_physics_object(self, *args, **kwargs):

        o_id = kwargs.get('o_id', None)
        if o_id is None:
            o_id: int = self.get_unique_id()
            kwargs['o_id'] = o_id

        commands = Dominoes.add_physics_object(self, *args, **kwargs)
        self.non_flex_objects.append(o_id)

        return commands

    # def add_physics_object(self,
    #                        record: ModelRecord,
    #                        position: Dict[str, float],
    #                        rotation: Dict[str, float],
    #                        mesh_expansion: float = 0,
    #                        particle_spacing: float = 0.125,
    #                        mass: float = 1,
    #                        scale: Optional[Dict[str, float]] = {"x": 1.0, "y": 1.0, "z": 1.0},                                      o_id: Optional[int] = None,
    #                        add_data: Optional[bool] = True,
    #                        **kwargs) -> List[dict]:

    #     mass_scale = mass

    #     commands = FlexDataset.add_solid_object(
    #         self,
    #         record = record,
    #         position = position,
    #         rotation = rotation,
    #         scale = scale,
    #         mesh_expansion = mesh_expansion,
    #         particle_spacing = particle_spacing,
    #         mass_scale = mass_scale,
    #         o_id = o_id)

    #     # TODO add data

    #     return commands


if __name__ == '__main__':
    import platform, os

    args = get_args("flex_dominoes")

    if platform.system() == 'Linux':
        if args.gpu is not None:
            os.environ["DISPLAY"] = ":0." + str(args.gpu)
        else:
            os.environ["DISPLAY"] = ":0"

        launch_build = False
    else:
        launch_build = True

    C = FlexDominoes(
        launch_build=launch_build,
        room=args.room,
        num_middle_objects=args.num_middle_objects,
        randomize=args.random,
        seed=args.seed,
        target_zone=args.zone,
        zone_location=args.zlocation,
        zone_scale_range=args.zscale,
        zone_color=args.zcolor,
        zone_material=args.zmaterial,
        zone_friction=args.zfriction,
        target_objects=args.target,
        probe_objects=args.probe,
        middle_objects=args.middle,
        target_scale_range=args.tscale,
        target_rotation_range=args.trot,
        probe_rotation_range=args.prot,
        probe_scale_range=args.pscale,
        probe_mass_range=args.pmass,
        target_color=args.color,
        probe_color=args.pcolor,
        middle_color=args.mcolor,
        collision_axis_length=args.collision_axis_length,
        force_scale_range=args.fscale,
        force_angle_range=args.frot,
        force_offset=args.foffset,
        force_offset_jitter=args.fjitter,
        force_wait=args.fwait,
        spacing_jitter=args.spacing_jitter,
        lateral_jitter=args.lateral_jitter,
        middle_scale_range=args.mscale,
        middle_rotation_range=args.mrot,
        middle_mass_range=args.mmass,
        horizontal=args.horizontal,
        remove_target=bool(args.remove_target),
        remove_zone=bool(args.remove_zone),
        ## not scenario-specific
        camera_radius=args.camera_distance,
        camera_min_angle=args.camera_min_angle,
        camera_max_angle=args.camera_max_angle,
        camera_min_height=args.camera_min_height,
        camera_max_height=args.camera_max_height,
        monochrome=args.monochrome,
        material_types=args.material_types,
        target_material=args.tmaterial,
        probe_material=args.pmaterial,
        middle_material=args.mmaterial,
        distractor_types=args.distractor,
        distractor_categories=args.distractor_categories,
        num_distractors=args.num_distractors,
        occluder_types=args.occluder,
        occluder_categories=args.occluder_categories,
        num_occluders=args.num_occluders,
        occlusion_scale=args.occlusion_scale,
        remove_middle=args.remove_middle,
        ramp_color=args.rcolor
    )

    if bool(args.run):
        C.run(num=args.num,
             output_dir=args.dir,
             temp_path=args.temp,
             width=args.width,
             height=args.height,
             write_passes=args.write_passes.split(','),
             save_passes=args.save_passes.split(','),
             save_movies=args.save_movies,
             save_labels=args.save_labels,
             args_dict=vars(args))
    else:
        end = C.communicate({"$type": "terminate"})
        print([OutputData.get_data_type_id(r) for r in end])
