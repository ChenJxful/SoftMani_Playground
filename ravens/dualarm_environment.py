#!/usr/bin/env python

import os
import sys
import time
import threading
import pkg_resources
import math
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

from ravens.gripper import Gripper, Suction, Robotiq2F85
from ravens import tasks, utils
from ravens import Environment
import copy


class DualArmEnvironment(Environment):
    def __init__(self, disp=False, hz=240):
        super().__init__(disp, hz)  # 基础的 bullet 环境
        self.primitives["pick_place_vessel"] = self.pick_place_vessel
        
    def camera_shoot(self):
        """
        use bullet camera
        """
        # show plt figure
        show_plot = True
        if show_plot:
            plt.figure()
            plt_im = plt.imshow(np.zeros((240,320,4)))
            plt.axis('off')
            plt.tight_layout(pad=0)
            
        while True:
            # camera position is set as the ee of the ur5_camera
            current_state = p.getLinkState(self.ur5_camera, 12, computeForwardKinematics=True)
            current_position = np.array(current_state[0])
            
            # camera target position is set as the ur5_2
            # ur5_2_ee_tip_link = 12 # if use gripper, ee_tip_link = 10
            ur5_2_ee_tip_link = 10
            ee1_state = p.getLinkState(self.ur5_2, ur5_2_ee_tip_link, computeForwardKinematics=True)
            target_position = list(ee1_state[0])
            
            view_mtx = p.computeViewMatrix(
            cameraEyePosition=list(current_position),
            cameraTargetPosition=target_position,
            cameraUpVector=[0, 0, 1]
            )
            
            proj_mtx=p.computeProjectionMatrixFOV(fov=60, aspect=640 / 480, nearVal=0.01, farVal=100)
            
            width = 400
            height = 400
            img = p.getCameraImage(width, height, view_mtx, proj_mtx)[2]
            
  
            if show_plot:
                plt_im.set_array(img)
                plt.gca().set_aspect(height/width)
                plt.draw()
                plt.pause(0.1)
                
    def camera_shoot_new_thread(self):
        self.shoot_thread = threading.Thread(target=self.camera_shoot, name="camera_shoot")
        self.shoot_thread.start()
        
    def camera_follow(self):
        """
        ur5_camera's ee follow ur5_2's, the orientation will not change 
        """
        # ur5_2_ee_tip_link = 12 # if use gripper, ee_tip_link = 10
        ur5_2_ee_tip_link = 10
        while True:
            time.sleep(0.1)
            ee1_state = p.getLinkState(self.ur5_2, ur5_2_ee_tip_link, computeForwardKinematics=True)
            current_state = p.getLinkState(self.ur5_camera, 12, computeForwardKinematics=True)
            
        
            track_position = list(ee1_state[0])
            track_position[1] += 0.6
            track_position[2] = max(0.1, track_position[2])
            
            track_position = np.array(track_position)
            
            current_position = np.array(current_state[0])
            
            diff = current_position-track_position
            diff_ = (np.sum(np.abs(diff)))
            if diff_>0.01:
                track_qut = np.array(p.getQuaternionFromEuler((0, 0, -0.5*math.pi)))
                track_pose = np.hstack((track_position, track_qut))
            
                success = self.movep(arm1_pose=None, arm2_pose=None, arm_camera_pose=track_pose)
                
    def camera_follow_new_thread(self):
        self.follow_thread = threading.Thread(target=self.camera_follow, name="camera_follow")
        self.follow_thread.start()
        
    def set_camPose(self, d=1.0, yaw=20, pitch=-35, target=(0.5, 0, 0.1)):
        p.resetDebugVisualizerCamera(
            cameraDistance=d,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target)  # 调整相机位置  TODO 好像不能调整焦距
        
        

    def reset(self, task, last_info=None, disable_render_load=True):
        '''初始化双机械臂环境
        '''
        # disable_render_load = False  # 初始化的时候允许渲染（耗时久）
        self.pause()
        self.task = task
        self.objects = []
        self.fixed_objects = []
        if self.use_new_deformable:  # default True
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        else:
            p.resetSimulation()

        p.setGravity(0, 0, -1)  # 手动修改重力
        # p.setGravity(0, 0, -9.8)

        self.set_camPose()

        # Slightly increase default movej timeout for the more demanding tasks.
        if self.is_bag_env():
            self.t_lim = 60
            if isinstance(self.task, tasks.names['bag-color-goal']):
                self.t_lim = 120

        # Empirically, this seems to make loading URDFs faster w/remote displays.
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        id_plane = p.loadURDF('assets/plane/plane.urdf', [0, 0, -0.001])
        id_ws = p.loadURDF('assets/ur5/workspace.urdf', [0.5, 0, 0])

        # Load UR5 robot arm equipped with task-specific end effector.
        self.ur5 = p.loadURDF(f'assets/ur5/ur5-{self.task.ee}.urdf', basePosition=(-0.1,0,0))
        ori_co = p.getQuaternionFromEuler((0, 0, math.pi))
        self.ur5_2 = p.loadURDF(f'assets/ur5/ur5-{self.task.ee}.urdf', basePosition=(1.1,0,0), baseOrientation=ori_co)
        
        ori_camera_arm = p.getQuaternionFromEuler((0, 0, -0.5*math.pi))
        self.ur5_camera = p.loadURDF(f'assets/ur5/ur5-suction.urdf', basePosition=(0.5,1,0), baseOrientation=ori_camera_arm)
        
        
        self.ee_tip_link_camera = 12
        if self.task.ee == 'suction':
            self.ee_tip_link = 12
            self.ee_tip_link_2 = 12
            self.ee = Suction(self.ur5, self.ee_tip_link-1, position=(0.387, 0.109, 0.351))
            self.ee_2 = Suction(self.ur5_2, self.ee_tip_link_2-1, position=(0.612, -0.109, 0.351))
            self.ee_camera = Suction(self.ur5_camera, self.ee_tip_link_camera-1, camera=True)
            
        elif self.task.ee == 'gripper':
            self.ee_tip_link = 10
            self.ee_tip_link_2 = 10
            self.ee = Robotiq2F85(self.ur5, self.ee_tip_link-1)
            self.ee_2 = Robotiq2F85(self.ur5_2, self.ee_tip_link_2-1)
            self.ee_camera = Suction(self.ur5_camera, self.ee_tip_link_camera-1, camera=True)

        else:
            self.ee = Gripper()

        # Get revolute joint indices of robot (skip fixed joints).
        utils.cprint('UR5 Arm1 setup...', 'blue')
        num_joints = p.getNumJoints(self.ur5)
        # print("num_joints:", num_joints)
        joints = [p.getJointInfo(self.ur5, i) for i in range(num_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]
        # print('self.joints:\n', '\n'.join([str(j) for j in joints]))

        utils.cprint('UR5 Arm2 setup...', 'blue')
        num_joints_2 = p.getNumJoints(self.ur5_2)
        # print("num_joints2:", num_joints_2)
        joints_2 = [p.getJointInfo(self.ur5_2, i) for i in range(num_joints_2)]
        self.joints_2 = [j[0] for j in joints_2 if j[2] == p.JOINT_REVOLUTE]
        # print('self.joints2:\n', '\n'.join([str(j) for j in joints_2]))
        
        utils.cprint('UR5 ArmCamera setup...', 'blue')
        num_joints_camera = p.getNumJoints(self.ur5_camera)
        # print("num_joints2:", num_joints_2)
        joints_camera = [p.getJointInfo(self.ur5_camera, i) for i in range(num_joints_camera)]
        self.joints_camera = [j[0] for j in joints_camera if j[2] == p.JOINT_REVOLUTE]
        # print('self.joints2:\n', '\n'.join([str(j) for j in joints_2]))

        # Move robot to home joint configuration.
        # print(len(self.joints))
        for i in range(len(self.joints)):
            p.resetJointState(self.ur5, self.joints[i], self.homej[i])
            p.resetJointState(self.ur5_2, self.joints_2[i], self.homej[i])
            p.resetJointState(self.ur5_camera, self.joints_camera[i], self.homej[i])
        
        # time.sleep(100)

        # Get end effector tip pose in home configuration.
        ee_tip_state = p.getLinkState(self.ur5, self.ee_tip_link)
        # print(ee_tip_state)
        self.home_pose = np.array(ee_tip_state[0] + ee_tip_state[1])
        ee_tip_state_2 = p.getLinkState(self.ur5_2, self.ee_tip_link_2)
        # print(ee_tip_state_2)
        self.home_pose_2 = np.array(ee_tip_state_2[0] + ee_tip_state_2[1])
        # time.sleep(100)

        # Reset end effector.
        self.ee.release()
        self.ee_2.release()
        self.ur5_list = [self.ur5, self.ee_tip_link, self.joints, self.ee]
        self.ur5_2_list = [self.ur5_2, self.ee_tip_link_2, self.joints_2, self.ee_2]
        self.ur5_camera_list = [self.ur5_camera, self.ee_tip_link_camera, self.joints_camera, self.ee_camera]
        

        # Seems like this should be BEFORE reset()
        # since for bag-items we may assign to True!
        task.exit_gracefully = False

        # Reset task. 重置血管
        if last_info is not None:
            task.reset(self, last_info)
        else:
            task.reset(self)

        # Daniel: might be useful to have this debugging tracker.
        self.IDTracker = utils.TrackIDs()
        self.IDTracker.add(id_plane, 'Plane')
        self.IDTracker.add(id_ws, 'Workspace')
        self.IDTracker.add(self.ur5, 'UR5')
        try:
            self.IDTracker.add(self.ee.body, 'Gripper.body')
        except:
            pass

        # Daniel: add other IDs, but not all envs use the ID tracker.
        try:
            task_IDs = task.get_ID_tracker()
            for i in task_IDs:
                self.IDTracker.add(i, task_IDs[i])
        except AttributeError:
            pass
        #print(self.IDTracker)  # If doing multiple episodes, check if I reset the ID dict!
        assert id_ws == 1, f'Workspace ID: {id_ws}'

        # Daniel: tune gripper for deformables if applicable, and CHECK HZ!!
        if self.is_softbody_env():  # default False
            self.ee.set_def_threshold(threshold=self.task.def_threshold)
            self.ee.set_def_nb_anchors(nb_anchors=self.task.def_nb_anchors)
            assert self.hz >= 480, f'Error, hz={self.hz} is too small!'

        # Restart simulation.
        self.start()
        if disable_render_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        (obs, _, _, _) = self.step()
        
        # time.sleep(100)
        
        self.camera_follow_new_thread()
        self.camera_shoot_new_thread()
        return obs
    
    def movep(self, arm1_pose, arm2_pose, arm_camera_pose, speed=0.01):
        """Move dual UR5s to target end effector pose."""
        # # Keep joint angles between -180/+180
        # targj[5] = ((targj[5] + np.pi) % (2 * np.pi) - np.pi)
        if arm1_pose is None:
            arm1_targj = None
        else:
            arm1_targj = self.solve_IK(self.ur5_list, arm1_pose)
            
        if arm2_pose is None:
            arm2_targj=None
        else:
            arm2_targj = self.solve_IK(self.ur5_2_list, arm2_pose)
            
            
        if arm_camera_pose is None:
            arm_camera_targj=None
        else:
            arm_camera_targj = self.solve_IK(self.ur5_camera_list, arm_camera_pose)
            
        return self.movej(arm1_targj, arm2_targj, arm_camera_targj, speed, self.t_lim)

    def solve_IK(self, arm, pose):
        '''add parameter 'arm'
        '''
        homej_list = np.array(self.homej).tolist()
        joints = p.calculateInverseKinematics(
            bodyUniqueId=arm[0],
            endEffectorLinkIndex=arm[1],
            targetPosition=pose[:3],
            targetOrientation=pose[3:],
            lowerLimits=[-17, -2.3562, -17, -17, -17, -17],
            upperLimits=[17, 0, 17, 17, 17, 17],
            jointRanges=[17] * 6,
            restPoses=homej_list,
            maxNumIterations=100,
            residualThreshold=1e-5)
        joints = np.array(joints)
        joints[joints > 2 * np.pi] = joints[joints > 2 * np.pi] - 2 * np.pi
        joints[joints < -2 * np.pi] = joints[joints < -2 * np.pi] + 2 * np.pi
        return joints
    
    def movej(self, arm1_targj, arm2_targj, arm_camera_targj, speed=0.01, t_lim=20):
        """Move dual UR5s to target joint configuration."""
        t0 = time.time()
        flag1 = False
        flag2 = False
        flag3 = False
        while (time.time() - t0) < t_lim:
            if arm1_targj is not None:
                arm1_currj = [p.getJointState(self.ur5_list[0], i)[0] for i in self.ur5_list[2]]
                arm1_currj = np.array(arm1_currj)
                arm1_diffj = arm1_targj - arm1_currj

                arm1_norm = np.linalg.norm(arm1_diffj)
                arm1_v = arm1_diffj / arm1_norm if arm1_norm > 0 else 0
                arm1_stepj = arm1_currj + arm1_v * speed + (np.random.random(arm1_v.shape) - 0.5) * 0.001  # add some noise
                arm1_gains = np.ones(len(self.ur5_list[2]))
                p.setJointMotorControlArray(
                    bodyIndex=self.ur5_list[0],
                    jointIndices=self.ur5_list[2],
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=arm1_stepj,
                    positionGains=arm1_gains)      
                
                if all(np.abs(arm1_diffj) < 1e-2):
                    flag1 = True
            else:
                flag1 = True
                
            if arm2_targj is not None:                       
                arm2_currj = [p.getJointState(self.ur5_2_list[0], i)[0] for i in self.ur5_2_list[2]]
                arm2_currj = np.array(arm2_currj)
                arm2_diffj = arm2_targj - arm2_currj
                
                arm2_norm = np.linalg.norm(arm2_diffj)
                arm2_v = arm2_diffj / arm2_norm if arm2_norm > 0 else 0
                arm2_stepj = arm2_currj + arm2_v * speed + (np.random.random(arm2_v.shape) - 0.5) * 0.001  # add some noise
                arm2_gains = np.ones(len(self.ur5_list[2]))
                p.setJointMotorControlArray(
                    bodyIndex=self.ur5_2_list[0],
                    jointIndices=self.ur5_2_list[2],
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=arm2_stepj,
                    positionGains=arm2_gains)
                
                if all(np.abs(arm2_diffj) < 1e-2):
                        flag2 = True
            else:
                flag2 = True
                
            if arm_camera_targj is not None:                       
                arm_camera_currj = [p.getJointState(self.ur5_camera_list[0], i)[0] for i in self.ur5_camera_list[2]]
                arm_camera_currj = np.array(arm_camera_currj)
                arm_camera_diffj = arm_camera_targj - arm_camera_currj
                
                arm_camera_norm = np.linalg.norm(arm_camera_diffj)
                arm_camera_v = arm_camera_diffj / arm_camera_norm if arm_camera_norm > 0 else 0
                arm_camera_stepj = arm_camera_currj + arm_camera_v * speed + (np.random.random(arm_camera_v.shape) - 0.5) * 0.001  # add some noise
                arm_camera_gains = np.ones(len(self.ur5_2_list[2]))
                p.setJointMotorControlArray(
                    bodyIndex=self.ur5_camera_list[0],
                    jointIndices=self.ur5_camera_list[2],
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=arm_camera_stepj,
                    positionGains=arm_camera_gains)
                
                if all(np.abs(arm_camera_diffj) < 1e-2):
                        flag3 = True
            else:
                flag3 = True
            
            
            
            time.sleep(0.001)
            if flag1 and flag2 and flag3:
                return True
        print('Warning: movej exceeded {} sec timeout. Skipping.'.format(t_lim))
        return False
    
    
    def pick_place_vessel(self, arm1_pose0, arm1_pose1, arm2_pose0, arm2_pose1):
        """Execute pick and place primitive.

        Standard ravens tasks use the `delta` vector to lower the gripper
        until it makes contact with something. With deformables, however, we
        need to consider cases when the gripper could detect a rigid OR a
        soft body (cloth or bag); it should grip the first item it touches.
        This is handled in the Gripper class.

        Different deformable ravens tasks use slightly different parameters
        for better physics (and in some cases, faster simulation). Therefore,
        rather than make special cases here, those tasks will define their
        own action parameters, which we use here if they exist. Otherwise, we
        stick to defaults from standard ravens. Possible action parameters a
        task might adjust:

            speed: how fast the gripper moves.
            delta_z: how fast the gripper lowers for picking / placing.
            prepick_z: height of the gripper when it goes above the target
                pose for picking, just before it lowers.
            postpick_z: after suction gripping, raise to this height, should
                generally be low for cables / cloth.
            preplace_z: like prepick_z, but for the placing pose.
            pause_place: add a small pause for some tasks (e.g., bags) for
                slightly better soft body physics.
            final_z: height of the gripper after the action. Recommended to
                leave it at the default of 0.3, because it has to be set high
                enough to avoid the gripper occluding the workspace when
                generating color/depth maps.
        Args:
            pose0: picking pose.
            pose1: placing pose.

        Returns:
            A bool indicating whether the action succeeded or not, via
            checking the sequence of movep calls. If any movep failed, then
            self.step() will terminate the episode after this action.
        """
        time.sleep(5)
        self.set_camPose(d=0.5, yaw=0, pitch=-20)
        print("arm1_pose0, arm1_pose1, arm2_pose0, arm2_pose1:", arm1_pose0, arm1_pose1, arm2_pose0, arm2_pose1)
        # Defaults used in the standard Ravens environments.
        speed = 0.01
        delta_z = -0.005
        prepick_z = 0.3
        postpick_z = 0.3
        preplace_z = 0.3
        pause_place = 0.0
        final_z = 0.3

        # Find parameters, which may depend on the task stage.
        if hasattr(self.task, 'primitive_params'):
            ts = self.task.task_stage
            if 'prepick_z' in self.task.primitive_params[ts]:
                prepick_z = self.task.primitive_params[ts]['prepick_z']
            speed       = self.task.primitive_params[ts]['speed']
            delta_z     = self.task.primitive_params[ts]['delta_z']
            postpick_z  = self.task.primitive_params[ts]['postpick_z']
            preplace_z  = self.task.primitive_params[ts]['preplace_z']
            pause_place = self.task.primitive_params[ts]['pause_place']

        # Used to track deformable IDs, so that we can get the vertices.
        def_IDs = []
        if hasattr(self.task, 'def_IDs'):
            def_IDs = self.task.def_IDs

        # Otherwise, proceed as normal.
        success = True
        arm1_pick_position = np.array(arm1_pose0[0])
        arm1_pick_rotation = np.array(arm1_pose0[1])
        arm1_prepick_position = arm1_pick_position.copy()
        arm1_prepick_position[2] = prepick_z
        
        arm2_pick_position = np.array(arm2_pose0[0])
        arm2_pick_rotation = np.array(arm2_pose0[1])
        arm2_prepick_position = arm2_pick_position.copy()
        arm2_prepick_position[2] = prepick_z

        # Execute picking motion primitive.
        arm1_prepick_pose = np.hstack((arm1_prepick_position, arm1_pick_rotation))
        arm2_prepick_pose = np.hstack((arm2_prepick_position, arm2_pick_rotation))
        
        success &= self.movep(arm1_prepick_pose, arm2_prepick_pose, arm_camera_pose=None)
        utils.cprint("Arrive prepick_pose", "yellow")
        
        arm1_target_pose = arm1_prepick_pose.copy()
        arm2_target_pose = arm2_prepick_pose.copy()
        delta = np.array([0, 0, delta_z, 0, 0, 0, 0])
        
        while True:
            if arm1_target_pose is not None:
                if not self.ee.detect_contact(def_IDs) and arm1_target_pose[2] > 0:  # what goes wrong here?
                    arm1_target_pose += delta
                else:
                    arm1_target_pose = None

            if arm2_target_pose is not None:
                if not self.ee_2.detect_contact(def_IDs) and arm2_target_pose[2] > 0:
                    arm2_target_pose += delta
                else:
                    arm2_target_pose = None

            if arm1_target_pose is None and arm2_target_pose is None:  # 存在可能：运动到0但是没有触碰到物体
                break
            else:   
                success &= self.movep(arm1_target_pose, arm2_target_pose, arm_camera_pose=None, speed=0.003)

        # Create constraint (rigid objects) or anchor (deformable).
        self.ee.activate(self.objects, def_IDs)
        self.ee_2.activate(self.objects, def_IDs)
        utils.cprint("Activate grasp", "yellow")

        # Increase z slightly (or hard-code it) and check picking success.
        arm1_prepick_pose[2] = 0.1
        arm2_prepick_pose[2] = 0.1
        arm1_prepick_pose[3:] = [0, 0, 0, 1]  # 摆正角度
        arm2_prepick_pose[3:] = [0, 0, 0, 1]
        success &= self.movep(arm1_prepick_pose, arm2_prepick_pose, arm_camera_pose=None, speed=0.003)
        utils.cprint("Raise up a little", "yellow")
        
        # self.set_camPose(d=0.3)s
        pick_success = self.ee.check_grasp() and self.ee_2.check_grasp()

        if pick_success:
            arm1_place_position = np.array(arm1_pose1[0])
            arm1_place_rotation = np.array(arm1_pose1[1])
            
            arm2_place_position = np.array(arm2_pose1[0])
            arm2_place_rotation = np.array(arm2_pose1[1])
            
            arm1_place_pose = np.hstack((arm1_place_position, arm1_place_rotation))
            arm2_place_pose = np.hstack((arm2_place_position, arm2_place_rotation))
            
            success &= self.movep(arm1_place_pose, arm2_place_pose, arm_camera_pose=None, speed=0.001)
            utils.cprint("Arrive place_pose", "yellow")

            time.sleep(2)

        else:
            utils.cprint("Grasp failed!", "red")
