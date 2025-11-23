import numpy as np
import roboverse.bullet as bullet

from .drawer_open_transfer import DrawerOpenTransfer
from .pick_place import PickPlace
from roboverse.envs.widow250_custom import Widow250DrawerRandomizedPickPlaceEnv as Env


class DrawerPickPlacePolicy:
    """Composite policy that first opens the drawer, then picks and places
    an object into the drawer using the existing scripted policies.

    Usage:
      policy = DrawerPickPlacePolicy(env)
      policy.reset()
      action, info = policy.get_action()
    """

    def __init__(self, env, **pick_kwargs):
        self.env: Env = env
        # instantiate subpolicies
        self.drawer_policy = DrawerOpenTransfer(env)
        # by default use PickPlace (grasp+place) policy
        self.pick_policy = PickPlace(env, **pick_kwargs)

        # parameters for the hang-away vertical motion
        self.hang_xyz_scale = 7.0
        self.hang_height_offset = 0.2  # how much above current to hang

        self.reset()

    def reset(self):
        # reset subpolicies and internal state
        try:
            self.drawer_policy.reset()
        except Exception:
            pass
        try:
            self.pick_policy.reset()
        except Exception:
            pass

        self.stage = 'open'  # stages: 'open' -> 'hang' -> 'pick'
        self.place_attempted = False
        self.hang_target_z = None

    def get_action(self):
        # Check environment info to guide stage transitions
        info = self.env.get_info()

        # If still opening stage, use drawer policy
        if self.stage == 'open':
            drawer_opened = bool(info.get('drawer_opened_success', False))
            if drawer_opened:
                self.stage = 'hang'
                # compute a hang target z using env ee_pos_init if available
                ee_pos, _ = bullet.get_link_state(self.env.robot_id, self.env.end_effector_index)
                self.hang_target_z = float(ee_pos[2]) + self.hang_height_offset
                stop_action = np.zeros(self.env.action_dim)
                return stop_action, dict(stage=self.stage, **info)
            else:
                action, a_info = self.drawer_policy.get_action()        
                agent_info = dict(stage=self.stage, **a_info)
                return action, agent_info

        # If we're in hang stage, produce a purely vertical upward motion
        if self.stage == 'hang':
            # get current ee position
            ee_pos, _ = bullet.get_link_state(self.env.robot_id, self.env.end_effector_index)
            z_diff = self.hang_target_z - float(ee_pos[2])
            # small tolerance to consider hang complete
            if z_diff <= 0.01:
                self.stage = 'pick'
                self.pick_policy.reset()
                stop_action = np.zeros(self.env.action_dim)
                return stop_action, dict(stage=self.stage, **info)
            else:
                # move vertically upwards only
                action_xyz = np.array([0.0, 0.0, z_diff]) * self.hang_xyz_scale
                action_angles = [0., 0., 0.]
                action_gripper = [0.0]
                neutral_action = [0.0]
                action = np.concatenate((action_xyz, action_angles, action_gripper, neutral_action))
                agent_info = dict(stage=self.stage, hang_target_z=self.hang_target_z)
                return action, agent_info
            
        if self.stage == 'pick':
            # Otherwise, use pick/place policy
            action, p_info = self.pick_policy.get_action()

            # update place_attempted flag if present in pick policy info
            if 'place_attempted' in p_info:
                self.place_attempted = bool(p_info['place_attempted'])

            agent_info = dict(
                stage=self.stage, **p_info)

            return action, agent_info
        
        raise RuntimeError(f"Unknown stage '{self.stage}' in DrawerPickPlacePolicy.")


class DrawerPickPlaceSuboptimal(DrawerPickPlacePolicy):
    def __init__(self, env, **pick_kwargs):
        super(DrawerPickPlaceSuboptimal, self).__init__(env, **pick_kwargs)
        # Replace subpolicies with suboptimal variants when available
        self.drawer_policy = DrawerOpenTransfer(env, suboptimal=True)
        try:
            self.pick_policy.reset()
        except Exception:
            pass
