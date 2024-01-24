from utils.MujocoRobot import *

#mjcf_path = "iCubGazeboV2_6.xml" , iRonCub-Mk1
mjcf_path = "mujoco_descriptions/iRonCub-Mk1.xml"
ergo = False
if "ergo" in mjcf_path:
    pos_0 = [0, 0, 0.0]
    ergo = True
elif "iRonCub" in mjcf_path:
    pos_0 = [0, 0, -0.1]
else:
    pos_0 = [0, 0, -0.1]

robot = MujocoRobot(mjcf_path, from_path=True)


l_fingures = ["l_ring_prox", "l_ring_dist", "l_index_add", "l_index_prox", "l_index_dist", "l_middle_prox", "l_middle_dist", "l_thumb_add", "l_thumb_prox", "l_thumb_dist", "l_pinkie_prox", "l_pinkie_dist"]
r_fingures = ["r_ring_prox", "r_ring_dist", "r_index_add", "r_index_prox", "r_index_dist", "r_middle_prox", "r_middle_dist", "r_thumb_add", "r_thumb_prox", "r_thumb_dist", "r_pinkie_prox", "r_pinkie_dist"]

with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
    mujoco.mj_resetDataKeyframe(robot.model, robot.data, 0)
    mujoco.mj_step(robot.model, robot.data)

    robot.print_contact_pair()

    robot.data.qpos[0:3] = pos_0

    # joint names to control
    print(robot.joint_names)

    wall = time.monotonic()
    with viewer.lock():
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0

    robot.set_ctrl("l_shoulder_roll", 1)
    robot.set_ctrl("r_shoulder_roll", 1)
    robot.set_ctrl("torso_pitch", 0.3)

    while viewer.is_running():
        t = time.monotonic() - wall

        # observe state
        # compute control
        st = np.sin(t) ** 2 * 0.1 + 0.1
        ct = np.cos(t) ** 2 * 0.1 + 0.1

        # set control
        robot.set_relative_ctrl(["l_shoulder_roll", "l_elbow"], [st, 2 * st])
        robot.set_relative_ctrl(["r_shoulder_roll", "r_elbow"], [st, 2 * st])
        robot.set_relative_ctrl(
            ["l_hip_pitch", "l_knee", "l_ankle_pitch"], [1.5 * st, -2 * st, -st]
        )
        robot.set_relative_ctrl(
            ["r_hip_pitch", "r_knee", "r_ankle_pitch"], [1.5 * st, -2 * st, -st]
        )

        # ergiCub has fingers
        if ergo:
            robot.set_relative_ctrl(l_fingures, [-2 * np.sin(2 * t)] * len(l_fingures))
            robot.set_relative_ctrl(r_fingures, [-2 * np.sin(2 * t)] * len(r_fingures))

        # step simulation
        while robot.data.time <= t:
            mujoco.mj_step(robot.model, robot.data)
        viewer.sync()
