from rlbench.backend.task import Task
from rlbench.backend.scene import DemoError
from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from rlbench.backend.const import TTT_FILE
from rlbench.backend.scene import Scene
from rlbench.backend.utils import task_file_to_task_class
from rlbench.backend.task import TASKS_PATH
from rlbench.backend.robot import Robot
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import TaskEnvironment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from pyrep.const import RenderMode
from rlbench.const import colors
import numpy as np
import os
import argparse
import json
from PIL import Image

DEMO_ATTEMPTS = 5
MAX_VARIATIONS = 100


class TaskValidationError(Exception):
    pass


def task_smoke(task: Task, scene: Scene, variation=-1, demos=200, success=0.50,
               max_variations=3, test_demos=True):
    print('Running task validator on task: %s' % task.get_name())

    # Loading
    scene.load(task)
    global count
    count = 0

    # Number of variations
    variation_count = task.variation_count()
    total_runs = int(demos // variation_count) if variation == -1 else demos
    if variation_count < 0:
        raise TaskValidationError(
            "The method 'variation_count' should return a number > 0.")

    if variation_count > MAX_VARIATIONS:
        raise TaskValidationError(
            "This task had %d variations. Currently the limit is set to %d" %
            (variation_count, MAX_VARIATIONS))

    # Base rotation bounds
    base_pos, base_ori = task.base_rotation_bounds()
    if len(base_pos) != 3 or len(base_ori) != 3:
        raise TaskValidationError(
            "The method 'base_rotation_bounds' should return a tuple "
            "containing a list of floats.")

    # Boundary root
    root = task.boundary_root()
    if not root.still_exists():
        raise TaskValidationError(
            "The method 'boundary_root' should return a Dummy that is the root "
            "of the task.")

    def variation_smoke(i):
        print('Running task validator on variation: %d' % i)

        attempt_result = False
        failed_demos = 0
        for j in range(DEMO_ATTEMPTS):
            failed_demos = run_demos(i)
            attempt_result = (failed_demos / float(demos) <= 1. - success)
            if attempt_result:
                break
            else:
                print('Failed on attempt %d. Trying again...' % j)

        # Make sure we don't fail too often
        if not attempt_result:
            raise TaskValidationError(
                "Too many failed demo runs. %d of %d demos failed." % (
                    failed_demos, demos))
        else:
            print('Variation %d of task %s is good!' % (i, task.get_name()))
            if test_demos:
                print('%d of %d demos were successful.' % (
                    total_runs - failed_demos, total_runs))

    def run_demos(variation_num):
        global count
        fails = 0
        for idx in range(total_runs):
            try:
                scene.reset()
                desc = scene.init_episode(variation_num, max_attempts=10)

                origin = task.boundary_root().get_position(), task.boundary_root().get_orientation()
                if idx == 0:
                    A_d, A_s= task.lightA.get_diffuse(), task.lightA.get_specular()
                    B_d, B_s= task.lightB.get_diffuse(), task.lightB.get_specular()
                    D_d, D_s= task.lightD.get_diffuse(), task.lightD.get_specular()

                light_cond = np.random.random()
                if light_cond > 0.8:
                    color_idx = np.random.randint(0, len(colors))
                    color_rgb = colors[color_idx][1]
                    light = np.random.choice([task.lightA, task.lightB, task.lightD], 1)[0]
                    light.set_diffuse(color_rgb)
                    light.set_specular(color_rgb)
                else:
                    task.lightA.set_diffuse(A_d); task.lightA.set_specular(A_s)
                    task.lightB.set_diffuse(B_d); task.lightB.set_specular(B_s)
                    task.lightD.set_diffuse(D_d); task.lightD.set_specular(D_s)
                if not isinstance(desc, list) or len(desc) <= 0:
                    raise TaskValidationError(
                        "The method 'init_variation' should return a list of "
                        "string descriptions.")
                if test_demos:
                    # inference with gt position
                    demo = scene.get_demo(record=True)
                    os.makedirs(f'./eval_logs/{task_name}/gt', exist_ok=True)
                    img = Image.fromarray(demo._observations[-1].wrist_rgb)
                    img.save(f'./eval_logs/{task_name}/gt/{count:03d}.png')
                    state_ = {'variation_num': count,
                                'left_gt_coord' : task.left.tolist(),
                                'left_gt_orientation': task.left_orient.tolist(),
                                'right_gt_coord': task.right.tolist(),
                                'right_gt_orientation': task.right_orient.tolist(),}
                    json_dump.append(state_)
            except DemoError as e:
                fails += 1
                print(e)
                continue
            except Exception as e:
                # TODO: check that we don't fall through all of these cases
                fails += 1
                print(e)

            scene.reset()
            desc = scene.init_episode(variation_num, max_attempts=10, randomly_place=False)
            
            if light_cond > 0.8:
                light.set_diffuse(color_rgb)
                light.set_specular(color_rgb)
            else:
                task.lightA.set_diffuse(A_d); task.lightA.set_specular(A_s)
                task.lightB.set_diffuse(B_d); task.lightB.set_specular(B_s)
                task.lightD.set_diffuse(D_d); task.lightD.set_specular(D_s)      
            task.boundary_root().set_position(origin[0])
            task.boundary_root().set_orientation(origin[1])
            task.augment()
            print(task.rot_aug, task.pos_aug)
            
            try:
                demo2 = scene.get_demo(record=True, randomly_place=False)
                os.makedirs(f'./eval_logs/{task_name}/aug', exist_ok=True)
                img = Image.fromarray(demo2._observations[-1].wrist_rgb)
                img.save(f'./eval_logs/{task_name}/aug/{count:03d}_aug1.png')

                state_ = json_dump[-1]
                json_dump.pop()
                state_['left_coord'] = task.left.tolist()
                state_['left_orientation'] = task.left_orient.tolist()
                state_['right_coord'] = task.right.tolist()
                state_['right_orientation'] = task.right_orient.tolist()
                state_['rot_aug'] = task.rot_aug
                state_['pos_aug'] = task.pos_aug
                json_dump.append(state_)
            except DemoError as e:
                fails += 1
                print(e)
                continue
            finally:
                count += 1
        return fails

    variations_to_test = range(variation_count) if variation == -1 else [variation]
    print(variations_to_test)
    # Task set-up
    scene.init_task()
    [variation_smoke(i) for i in variations_to_test]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="The task file to test.")
    parser.add_argument("--headless", action='store_true')
    parser.add_argument("--collision_checking", action='store_true')
    args = parser.parse_args()

    python_file = os.path.join(TASKS_PATH, args.task)
    if not os.path.isfile(python_file):
        raise RuntimeError('Could not find the task file: %s' % python_file)

    task_class = task_file_to_task_class(args.task)
    sim = PyRep()
    ttt_file = os.path.join('/home/commonsense/data/cvpr/3d_diffuser_actor/RLBench/rlbench', TTT_FILE)

    sim.launch(ttt_file, headless=args.headless, responsive_ui=True)
    sim.step_ui()
    sim.set_simulation_timestep(0.005)
    sim.step_ui()
    sim.start()
    robot = Robot(Panda(), PandaGripper())
    active_task = task_class(sim, robot)
    task_name = active_task.get_name()

    # camera
    obs = ObservationConfig()
    obs.set_all(False)
    cam_config = CameraConfig(rgb=True, depth=False, mask=False,
                              render_mode=RenderMode.OPENGL)
    obs.wrist_camera = cam_config
    scene = Scene(sim, robot, obs)

    os.makedirs('./eval_logs', exist_ok=True)
    os.makedirs(f'./eval_logs/{task_name}', exist_ok=True)
    json_dump = []
    try:
        task_smoke(active_task, scene, variation=-1)
    except TaskValidationError as e:
        sim.shutdown()
        raise e
    sim.shutdown()
    
    # post-process
    tmp_lst = []
    for i, case in enumerate(json_dump):
        try:
            case['left_coord']
        except KeyError:
            tmp_lst.append(i)
    
    json_dump = [json_dump[i] for i in range(len(json_dump)) if i not in tmp_lst]
    for directory in ['gt', 'aug']:
        for filename in os.listdir(f'./eval_logs/{task_name}/{directory}'):
            if any(str(index) in filename for index in tmp_lst):
                os.remove(f'./eval_logs/{task_name}/{directory}/{filename}')

    with open(f'./eval_logs/{task_name}/log.json', 'w') as f:
        json.dump(json_dump, f, indent=4)
    print('Validation successful!')
