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
import cv2
from ravens.gripper import Gripper, Suction, Robotiq2F85
from ravens import tasks, utils
from ravens import Environment
import copy

from pybullet_planning import load_pybullet, connect, wait_for_user, LockRenderer, has_gui, WorldSaver, HideOutput, \
    reset_simulation, disconnect, set_camera_pose, has_gui, set_camera, wait_for_duration, wait_if_gui, apply_alpha
from pybullet_planning import Pose, Point, Euler
from pybullet_planning import multiply, invert, get_distance
from pybullet_planning import create_obj, create_attachment, Attachment, get_grasp_pose
from pybullet_planning import link_from_name, get_link_pose, get_moving_links, get_link_name, get_disabled_collisions, \
    get_body_body_disabled_collisions, has_link, are_links_adjacent
from pybullet_planning import get_num_joints, get_joint_names, get_movable_joints, get_joint_positions, set_joint_positions, joint_from_name, \
    joints_from_names, get_sample_fn, plan_joint_motion, check_initial_end

class DualArmEnvironment(Environment):
    def __init__(self, disp=False, hz=240):
        super().__init__(disp, hz)  # 基础的 bullet 环境
        self.primitives["pick_place_vessel"] = self.pick_place_vessel
        self.find_target = False
        self.target_points = [] # each element is a list [x, y, z, qx, qy, qz, qw]

    def camera_shoot(self):
        """
        use bullet camera
        """
        # show plt figure
        show_plot = True
        width = 640
        height = 480
        if show_plot:
            plt.figure()
            plt_im = plt.imshow(np.zeros((height,width,4)))
            plt.axis('off')
            plt.tight_layout(pad=0)
            
        while True:
            # camera position is set as the ee of the ur5_camera
            current_state = p.getLinkState(self.ur5_camera, 12, computeForwardKinematics=True)
            current_position = np.array(current_state[0])
            # print("current camera position:", current_position)

            # camera target position is set as the ur5_2
            # if use suction, ur5_2_ee_tip_link = 12. if use gripper, ee_tip_link = 10
            ur5_2_ee_tip_link = 10
            ee2_state = p.getLinkState(self.ur5_2, ur5_2_ee_tip_link, computeForwardKinematics=True)
            target_position = list(ee2_state[0])

            # 用于设置相机的位置，此处选择俯瞰水平面
            current_position = [0.5, 0.3, 0.5]
            target_position = [0.5,0.3001, 0] # 设置为 0.3 会导致无法看到物体，不知原因
            view_mtx = p.computeViewMatrix(
            cameraEyePosition=list(current_position),
            cameraTargetPosition=target_position,
            cameraUpVector=[0, 0, 1]
            )
            # print("view_mtx:", view_mtx)
            
            proj_mtx=p.computeProjectionMatrixFOV(fov=69.4, aspect=width / height, nearVal=0.2, farVal=90)
            # print("proj_mtx:", proj_mtx)

            
            # 读取摄像头数据
            img = p.getCameraImage(width, height, view_mtx, proj_mtx)[2]
            # 存储图像为 vessel.npy
            # np.save("figures/graduation_design/vessel.npy", img)

            ############
            # show img #
            ### START ##
            # width, height = img.shape[1], img.shape[0]
            # img = img[:, :, :3] # rgb img
            # plt_im.set_array(img)
            # plt.gca().set_aspect(height/width)
            # plt.draw()
            # plt.savefig("figures/camera.png", bbox_inches='tight', pad_inches=0)
            # time.sleep(100)
            # plt.pause(100)
            #### END ####

            # 最开始要寻找目标点
            target_pixel_points = [] # used for test
            if not self.find_target:
                utils.cprint("Finding target points...", "green")
                width, height = img.shape[1], img.shape[0]
                img = img[:, :, :3] # rgb img
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # convert to bgr
                processed_image, center_coords, end_points_pairs = utils.process_image(img)

                if len(center_coords) != 2:
                    utils.cprint("Failed to find target points.", "red")
                    continue

                # calculate the target pos in world frame
                Z_c = -current_position[2] + 0.01
                for i, center in enumerate(center_coords):
                    world_point_pos = utils.get_world_pos_from_pixel_pos(center, np.array(view_mtx).reshape((4,4), order='F'), np.array(proj_mtx).reshape((4,4), order='F'), width, height, Z_c)
                    end_pair = end_points_pairs[i]
                    p1 = utils.get_world_pos_from_pixel_pos(end_pair[0], np.array(view_mtx).reshape((4,4), order='F'), np.array(proj_mtx).reshape((4,4), order='F'), width, height, Z_c)
                    p2 = utils.get_world_pos_from_pixel_pos(end_pair[1], np.array(view_mtx).reshape((4,4), order='F'), np.array(proj_mtx).reshape((4,4), order='F'), width, height, Z_c)
                    target_vector = p2 - p1
                    
                    ref_vector = [1, 0, 0]
                    quarternion = utils.calculate_rotation_quaternion_from_vectors(ref_vector, target_vector)
                    # print("euler:", p.getEulerFromQuaternion(quarternion))
                    self.target_points.append(np.concatenate((world_point_pos, quarternion)))

                if self.target_points[0][0] > self.target_points[1][0]:
                    # leave the left point(smaller x value) in the first place
                    self.target_points[0], self.target_points[1] = self.target_points[1], self.target_points[0]
                
                # utils.cprint("Successfully found target points.", "green")
                # print(self.target_points)
                self.find_target = True


            print("=======================")
            target_points_true = [(0.36076446339821966, 0.14370882364984666, 0.009989261182071722, 1), 
                             (0.6375725413462177, 0.05771920072101908, 0.009988209741870424, 1)]
            target_pixel_points = []
            # 通过 target point 在世界坐标系中的位置计算抓取点在像素坐标系中的位置
            for target_point in target_points_true:
                pixel_point = utils.get_pixel_pos_from_world_pos(target_point, np.array(view_mtx).reshape((4,4), order='F'), np.array(proj_mtx).reshape((4,4), order='F'), width, height)
                target_pixel_points.append(pixel_point)
                print("pixel_point:", pixel_point)

            print("-----------------------")
            # 通过像素坐标系中的位置计算抓取点在世界坐标系中的位置

            # test
            Z_c = -current_position[2] + 0.01
            for (i, pixel_point) in enumerate(target_pixel_points):
                world_point = utils.get_world_pos_from_pixel_pos(pixel_point, np.array(view_mtx).reshape((4,4), order='F'), np.array(proj_mtx).reshape((4,4), order='F'), width, height, Z_c)
                print("world_point:", world_point)

                # 分别计算 x， y， z 的误差绝对值
                x_error = abs(world_point[0] - target_points_true[i][0])
                y_error = abs(world_point[1] - target_points_true[i][1])
                z_error = abs(world_point[2] - target_points_true[i][2])
                # print("x_error, y_error, z_error:", x_error, y_error, z_error)
                # 输出误差的均方标准差
                print("MSE:", np.sqrt(np.mean(np.square([x_error, y_error, z_error]))))
            # test done.

            if show_plot:
                plt_im.set_array(img)
                plt.gca().set_aspect(height/width)
                plt.draw()
                # plt.savefig("figures/camera.png", bbox_inches='tight', pad_inches=0)
                # time.sleep(100)
                plt.pause(0.1)
                
    def camera_shoot_new_thread(self):
        self.shoot_thread = threading.Thread(target=self.camera_shoot, name="camera_shoot")
        self.shoot_thread.start()
        
    def camera_follow(self):
        """
        ur5_camera's ee follow ur5_2's, the orientation will not change 
        """
        # ur5_2_ee_tip_link = 12 # if use gripper, ee_tip_link = 10
        # ur5_2_ee_tip_link = 10
        # while True:
        #     time.sleep(0.1)
        #     ee2_state = p.getLinkState(self.ur5_2, ur5_2_ee_tip_link, computeForwardKinematics=True)
        #     current_state = p.getLinkState(self.ur5_camera, 12, computeForwardKinematics=True)
            
        
        #     track_position = list(ee2_state[0])
        #     track_position[1] += 0.6
        #     track_position[2] = max(0.1, track_position[2])
            
        #     track_position = np.array(track_position)
            
        #     current_position = np.array(current_state[0])
            
        #     diff = current_position-track_position
        #     diff_ = (np.sum(np.abs(diff)))
        #     if diff_>0.01:
        #         track_qut = np.array(p.getQuaternionFromEuler((0, 0, -0.5*math.pi)))
        #         track_pose = np.hstack((track_position, track_qut))
            
        #         success = self.movep(arm1_pose=None, arm2_pose=None, arm_camera_pose=track_pose)
        pass
                
    def camera_follow_new_thread(self):
        self.follow_thread = threading.Thread(target=self.camera_follow, name="camera_follow")
        self.follow_thread.start()
        
    def set_camPose(self, d=1, yaw=20, pitch=-40, target=(0.5, 0., 0.)):
        p.resetDebugVisualizerCamera(
            cameraDistance=d,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target)  # 调整相机位置  TODO 好像不能调整焦距
    
    def add_obstacles(self):
        """
        Asks the user whether to add obstacles to the simulation environment.
        If the user inputs 'y', it proceeds to add obstacles.
        If the user inputs 'n', it exits the function.
        If the input is neither, it repeats the question.
        """
        while True:
            user_input = input("Do you want to add obstacles? (y or n): ")
            if user_input.lower() == 'y':
                obstacle_positions = [[0.1, 0, 0.2], [0.7, -0.2, 0.2], [0.7, -0.1, 0.45], [0.7, -0.2, 0.45], [0.7, 0, 0.45], [0.5, -0.1, 0.35], [0.5, -0.1, 0.45]]
                print("Adding obstacles...")
                obstacles = []
                for obstacle_position in obstacle_positions:
                    cube_id = p.loadURDF('ravens/assets/cubes/cube_small.urdf', basePosition=obstacle_position)
                    # 创建一个固定关节，将立方体固定在空中
                    constraintId = p.createConstraint(parentBodyUniqueId=cube_id,
                                                    parentLinkIndex=-1,
                                                    childBodyUniqueId=-1,
                                                    childLinkIndex=-1,
                                                    jointType=p.JOINT_FIXED,
                                                    jointAxis=(0, 0, 0),
                                                    parentFramePosition=(0, 0, 0),
                                                    childFramePosition=obstacle_position)
                    obstacles.append(cube_id)
                return obstacles 
            elif user_input.lower() == 'n':
                None
            else:
                print("Invalid input, please enter 'y' for yes or 'n' for no.")

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

        self.id_plane = p.loadURDF('assets/plane/plane.urdf', [0, 0, -0.0001]) # inorder to show workspace correctly
        self.id_ws = p.loadURDF('assets/ur5/workspace.urdf', [0.5, 0, 0])

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
        # self.IDTracker = utils.TrackIDs()
        # self.IDTracker.add(id_plane, 'Plane')
        # self.IDTracker.add(id_ws, 'Workspace')
        # self.IDTracker.add(self.ur5, 'UR5')
        # try:
        #     self.IDTracker.add(self.ee.body, 'Gripper.body')
        # except:
        #     pass

        # Daniel: add other IDs, but not all envs use the ID tracker.
        # try:
        #     task_IDs = task.get_ID_tracker()
        #     for i in task_IDs:
        #         self.IDTracker.add(i, task_IDs[i])
        # except AttributeError:
        #     pass
        #print(self.IDTracker)  # If doing multiple episodes, check if I reset the ID dict!
        assert self.id_ws == 1, f'Workspace ID: {self.id_ws}'

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

        self.obstacles = self.add_obstacles()
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
                arm1_stepj = arm1_currj + arm1_v * speed + (np.random.random(arm1_v.shape) - 0.5) * 0.00001  # add some noise
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
                arm2_stepj = arm2_currj + arm2_v * speed + (np.random.random(arm2_v.shape) - 0.5) * 0.00001  # add some noise
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
                arm_camera_stepj = arm_camera_currj + arm_camera_v * speed + (np.random.random(arm_camera_v.shape) - 0.5) * 0.0001  # add some noise
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
        self.set_camPose(d=0.7)
        # print("arm1_pose0, arm1_pose1, arm2_pose0, arm2_pose1:", arm1_pose0, arm1_pose1, arm2_pose0, arm2_pose1)
        # Defaults used in the standard Ravens environments.
        speed = 0.01
        delta_z = -0.001
        prepick_z = 0.24
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

        # utils.cprint(def_IDs, "red")

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
        
        #TODO: 给机械臂2加入轨迹规划模块
        success &= self.movep(arm1_prepick_pose, arm2_pose=None, arm_camera_pose=None)

        attachment2 = Attachment(self.ur5_2, self.ee_tip_link_2-1, get_grasp_pose(self.ee_2.base2ur5_cons), self.ee_2.body)
        end_conf2 = self.solve_IK(self.ur5_2_list, arm2_prepick_pose)

        wait_for_user('Press enter to start planning!')
        path = plan_joint_motion(self.ur5_2, self.ur5_2_list[2], end_conf2, obstacles=self.obstacles+[self.id_plane], attachments=[attachment2], self_collisions=False, verbose=True)
        if path is None:
            utils.cprint('no plan found', 'red')
            wait_for_user()
        else:
            wait_for_user('a motion plan is found! Press enter to start simulating!')

        # move to init pose
        success &= self.movep(arm1_prepick_pose, arm2_pose=None, arm_camera_pose=None)

        # # adjusting this number will adjust the simulation speed
        time_step = 0.3 if len(path) < 10 else 0.03
        print(len(path))
        for conf in path:
            set_joint_positions(self.ur5_2, self.ur5_2_list[2], conf)
            attachment2.assign()
            wait_for_duration(time_step)


        utils.cprint("Arrive prepick_pose", "yellow")
        
        # time.sleep(200)

        arm1_target_pose = arm1_prepick_pose.copy()
        arm2_target_pose = arm2_prepick_pose.copy()
        delta = np.array([0, 0, delta_z, 0, 0, 0, 0])
        
        while True:
            if arm1_target_pose is not None:
                if not self.ee.detect_contact() and arm1_target_pose[2] > 0.003:  # what goes wrong here?
                    arm1_target_pose += delta
                else:
                    arm1_target_pose = None

            if arm2_target_pose is not None:
                if not self.ee_2.detect_contact() and arm2_target_pose[2] > 0:
                    arm2_target_pose += delta
                else:
                    arm2_target_pose = None

            if arm1_target_pose is None and arm2_target_pose is None:  # 存在可能：运动到0但是没有触碰到物体
                # raise up a little
                # ee1_state = p.getLinkState(self.ur5, self.ee_tip_link)
                # arm1_target_pose = np.hstack((ee1_state[0], ee1_state[1])) - 4*delta
                # ee2_state = p.getLinkState(self.ur5_2, self.ee_tip_link_2)
                # arm2_target_pose = np.hstack((ee2_state[0], ee2_state[1])) - 4*delta
                # print(arm1_target_pose, arm2_target_pose)
                # self.movep(arm1_target_pose, arm2_target_pose, arm_camera_pose=None, speed=0.003)
                # utils.cprint("Raise up a little", "yellow")
                break
            else:   
                success &= self.movep(arm1_target_pose, arm2_target_pose, arm_camera_pose=None, speed=0.01)
        # if arm1_target_pose[2] > 0.05:
        #     arm1_target_pose += delta
        #     arm2_target_pose += delta
        #     self.movep(arm1_target_pose, arm2_target_pose, arm_camera_pose=None, speed=0.003)
        # else:
        #     break

        # utils.cprint("Arrive pick_pose", "yellow")

        # time.sleep(200)

        # Create constraint (rigid objects) or anchor (deformable).
        self.ee.activate(self.objects)
        self.ee_2.activate(self.objects)

        # wait for the gripper to close
        for _ in range(200):
            p.stepSimulation()
            time.sleep(1/240)

        utils.cprint("Activate grasp", "yellow")
        time.sleep(0.01)
        # time.sleep(200)

        # Increase z slightly (or hard-code it) and check picking success.
        arm1_prepick_pose[2] = 0.25
        arm2_prepick_pose[2] = 0.25
        arm1_prepick_pose[3:] = arm1_pick_rotation
        arm2_prepick_pose[3:] = arm2_pick_rotation
        success &= self.movep(arm1_prepick_pose, arm2_prepick_pose, arm_camera_pose=None, speed=0.003)
        utils.cprint("Raise up a little", "yellow")
        
        # self.set_camPose(d=0.3)s
        pick_success = self.ee.check_grasp() and self.ee_2.check_grasp()
        
        if pick_success:
            utils.cprint("Successfully picked.", "yellow")
            arm1_place_position = np.array(arm1_pose1[0])
            arm1_place_position[2] = 0.3
            arm1_place_rotation = np.array(arm1_pose1[1])
            # arm1_place_rotation = p.getQuaternionFromEuler((0, 0, 0.5*np.pi))
            
            arm2_place_position = np.array(arm2_pose1[0])
            arm2_place_position[2] = 0.3
            arm2_place_rotation = np.array(arm2_pose1[1])
            # arm2_place_rotation = p.getQuaternionFromEuler((0, 0, -0.5*np.pi))
            
            arm1_place_pose = np.hstack((arm1_place_position, arm1_place_rotation))
            arm2_place_pose = np.hstack((arm2_place_position, arm2_place_rotation))
            
            success &= self.movep(arm1_place_pose, arm2_place_pose, arm_camera_pose=None, speed=0.006)
            utils.cprint("Arrive place_pose", "yellow")
            self.pause()
            time.sleep(5)

        else:
            utils.cprint("Grasp failed!", "red")
            self.stop()
    
