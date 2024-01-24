import argparse as ap
from utils.MujocoRobot import *

icub_path = "robot_descriptions/icub-models/iCub/"
ergocub_path = "robot_descriptions/ergocub-software/urdf/ergoCub/"

icub_names = ["iCubGazeboV2_6", "iCubGazeboV2_5", "iCubGazeboV2_7", "iCubGenova09"]
ergocub_names = ["ergoCubGazeboV1", "ergoCubGazeboV1_1", "ergoCubSN000", "ergoCubSN001"]

icub_descriptions = {model: icub_path+"robots/{}/model.urdf".format(model) for model in icub_names}
ergocub_descriptions = {model: ergocub_path+"robots/{}/model.urdf".format(model) for model in ergocub_names}

if __name__ == '__main__':
    args = ap.ArgumentParser()
    args.add_argument('--model', type=str, default="iCubGazeboV2_6", help="model name")
    args.add_argument('--kp', type=float, default=500, help="joint stiffness")
    args.add_argument('--damping', type=float, default=50.0, help="joint damping")
    args.add_argument('--add_floating', type=bool, default=True)
    args.add_argument('--launch-viewer',type=bool, default=True)

    args = args.parse_args()
    
    print("iCub models:")
    for key in icub_names:
        print(key, end=" ")
    print("ergoCub models:")
    for key in icub_names:
        print(key, end=" ")


    if args.model in icub_descriptions.keys():
        filepath = icub_descriptions[args.model]
        meshdir = icub_path + "meshes/simmechanics"

    elif args.model in ergocub_descriptions.keys():
        filepath = ergocub_descriptions[args.model]
        meshdir = ergocub_path + "meshes/simmechanics"
    else:
        raise ValueError("model not found")
    
    xml_string = generate_mjcf_scene(filepath, meshdir, root_link="root_link", kp=args.kp, damping=args.damping, add_floating=args.add_floating)

    to_save = update_meshdir_xml_str(xml_string, "../"+meshdir)

    with open("mujoco_descriptions/{}.xml".format(args.model), "w") as f:
        f.write(to_save)

    # reload with scene
    robot = MujocoRobot(xml_string, from_path=False)
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        mujoco.mj_resetDataKeyframe(robot.model, robot.data, 0)
        mujoco.mj_step(robot.model, robot.data)

        robot.print_contact_pair()

        wall = time.monotonic()
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0

        while viewer.is_running():
            t = time.monotonic() - wall

            while (robot.data.time <= t):
                mujoco.mj_step(robot.model, robot.data)
            viewer.sync()
            