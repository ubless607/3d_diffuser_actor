from typing import List
import numpy as np
from pyrep.objects import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, ConditionSet, GraspedCondition, OrConditions, GripperTouchCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.const import colors

class InsertOntoSquarePeg(Task):

    def init_task(self) -> None:
        self._square_ring = Shape('square_ring')
        self._success_centre = Dummy('success_centre')
        success_detectors = [ProximitySensor(
            'success_detector%d' % i) for i in range(4)]
        touch_detectors = [ProximitySensor(
            'touch_detector%d' % i) for i in range(4)]
        self.register_graspable_objects([self._square_ring])
        self.success_condition = [GripperTouchCondition(self.robot.gripper, touch_detectors[0]), GripperTouchCondition(self.robot.gripper, touch_detectors[2])]
        #success_condition = ConditionSet([DetectedCondition(
        #   self._square_ring, sd) for sd in success_detectors])
        self.register_success_conditions([OrConditions(self.success_condition)])

    def init_episode(self, index: int) -> List[str]:
        for cond in self.success_condition:
            cond.reset()
        
        color_name, color_rgb = colors[index]
        spokes = [Shape('pillar0'), Shape('pillar1'), Shape('pillar2')]
        chosen_pillar = np.random.choice(spokes)
        chosen_pillar.set_color(color_rgb)
        _, _, z = self._success_centre.get_position()
        x, y, _ = chosen_pillar.get_position()
        self._success_centre.set_position([x, y, z])

        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)
        spokes.remove(chosen_pillar)
        for spoke, i in zip(spokes, color_choices):
            name, rgb = colors[i]
            spoke.set_color(rgb)
        b = SpawnBoundary([Shape('boundary0')])
        b.sample(self._square_ring)
        return ['put the ring on the %s spoke' % color_name,
                'slide the ring onto the %s colored spoke' % color_name,
                'place the ring onto the %s spoke' % color_name]

    def variation_count(self) -> int:
        return len(colors)