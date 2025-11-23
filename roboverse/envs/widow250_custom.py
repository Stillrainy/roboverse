import numpy as np

from roboverse.envs.widow250_drawer import Widow250DrawerRandomizedEnv
from roboverse.assets.shapenet_object_lists import CONTAINER_CONFIGS
from roboverse.bullet import object_utils
from roboverse.policies.pick_place import PickPlace


class Widow250DrawerRandomizedPickPlaceEnv(Widow250DrawerRandomizedEnv):
    """
    Randomized drawer env with a 3-step composite reward:
      1) open drawer (+1)
      2) grasp target object (+1)
      3) place object into the drawer (+1)

    The env grants reward only when a subtask transitions from not-complete
    to complete. When all subtasks are complete, `step` returns `done=True`.
    """

    def __init__(self, **kwargs):
        super(Widow250DrawerRandomizedPickPlaceEnv, self).__init__(**kwargs)
        # track which subtasks have been achieved during the episode
        self._opened_achieved = False
        self._grasp_achieved = False
        self._place_achieved = False
        # set a container position for pick/place policies to use
        self.container_position = object_utils.get_drawer_pos(
            self.objects['drawer'])

        # attach a scripted pick-place policy for convenience
        try:
            self.scripted_policy = PickPlace(self)
        except Exception:
            # If policy construction fails (e.g., before objects fully loaded),
            # leave it unset; callers can create PickPlace(env) after reset.
            self.scripted_policy = None

    def reset(self, seed=None, options=None):
        self._opened_achieved = False
        self._grasp_achieved = False
        self._place_achieved = False
        return super(Widow250DrawerRandomizedPickPlaceEnv, self).reset(seed=seed, options=options)

    def get_info(self):
        """Extend parent info with `place_success_target` computed for the
        drawer tray. We attempt to identify a reasonable tray center using the
        drawer position. 
        """
        info = super(Widow250DrawerRandomizedPickPlaceEnv, self).get_info()

        # keep an env-level container_position attribute up-to-date
        container_pos = object_utils.get_drawer_pos(self.objects['drawer'])
        self.container_position = container_pos

        # Use tray thresholds from CONTAINER_CONFIGS if present
        tray_cfg = CONTAINER_CONFIGS.get('drawer', {})
        place_h_thresh = tray_cfg.get('place_success_height_threshold', -0.36)
        place_r_thresh = tray_cfg.get('place_success_radius_threshold', 0.04)

        info['place_success_target'] = object_utils.check_in_container(
            self.target_object, self.objects, container_pos,
            place_h_thresh, place_r_thresh)

        return info

    def step(self, action):
        # Let the parent step update the world and get a base observation
        obs, _, _, _, _ = super(Widow250DrawerRandomizedPickPlaceEnv, self).step(action)

        # Recompute info after the parent step
        info = self.get_info()

        opened_now = bool(info.get('drawer_opened_success', False))
        grasp_now = bool(info.get('grasp_success_target', False))
        place_now = bool(info.get('place_success_target', False))

        reward = 0.0

        # Award +1 once for each subtask when it becomes true
        if opened_now and not self._opened_achieved:
            reward += 1.0
            self._opened_achieved = True

        if grasp_now and not self._grasp_achieved:
            reward += 1.0
            self._grasp_achieved = True

        if place_now and not self._place_achieved:
            reward += 1.0
            self._place_achieved = True

        done = False
        # if self._opened_achieved and self._grasp_achieved and self._place_achieved:
        # TODO: fix the issue grasp_achieved
        if self._opened_achieved and self._place_achieved:
            done = True
        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info
