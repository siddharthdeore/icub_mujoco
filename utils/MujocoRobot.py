import time
import numpy as np
import os
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer


def extract_mj_names(model, num_obj, obj_type):
    id2name = {i: None for i in range(num_obj)}
    name2id = {}

    for i in range(num_obj):
        name = mujoco.mj_id2name(model, obj_type, i)
        name2id[name] = i
        id2name[i] = name

    return [id2name[nid] for nid in sorted(name2id.values())], name2id, id2name

class MujocoRobot:
    def __init__(self, xml_path_str, position=[0, 0, 0], orientation=[1.0, 0, 0, 0], from_path=True):
        self.xml_path_str = xml_path_str
        if(from_path):
            self.model = mujoco.MjModel.from_xml_path(xml_path_str)
            self.data = mujoco.MjData(self.model)
            self._load(position, orientation)
        else:
            self.model = mujoco.MjModel.from_xml_string(xml_path_str)
            self.data = mujoco.MjData(self.model)
            self._load(position, orientation)

    def _load(self, position, orientation):

        self.nq = self.model.nq
        self.link_names = []
        self.joint_names = []
        m = self.model
        (
            self.body_names,
            self._body_name2id,
            self._body_id2name,
        ) = extract_mj_names(m, m.nbody, mujoco.mjtObj.mjOBJ_BODY)
        (
            self.joint_names,
            self._joint_name2id,
            self._joint_id2name,
        ) = extract_mj_names(m, m.njnt, mujoco.mjtObj.mjOBJ_JOINT)
        (
            self.actuator_names,
            self._actuator_name2id,
            self._actuator_id2name,
        ) = extract_mj_names(m, m.nu, mujoco.mjtObj.mjOBJ_ACTUATOR)
        (
            self.sensor_names,
            self._sensor_name2id,
            self._sensor_id2name,
        ) = extract_mj_names(m, m.nsensor, mujoco.mjtObj.mjOBJ_SENSOR)

        self.actuators = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if(name is not None):
                # self.joint_names.append(name)
                self.actuators[name] = i
                #print(f"mujoco actuator {i} {name}")

        for i in range(self.nq):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if(name is not None):
                self.link_names.append(name)

        self.data.qpos[0:3] = position
        self.data.qpos[3:7] = orientation

        self.revolute_jnt_index = np.where(
            self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
        )[0].astype(np.int32)
        self.floating_jnt_index = np.where(
            self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
        )[0].astype(np.int32)
        #self.set_ctrl_from_keyframe()
        self.init_ctrl_ref = {}

        itr = 0
        if(len(self.data.ctrl)>0):
            for i in self.revolute_jnt_index:
                self.init_ctrl_ref[self._joint_id2name[i]] = self.data.ctrl[itr]
                itr = itr + 1

    def set_relative_ctrl(self, name, val):
        # set relative cotrol signal with respect to initial value
        if isinstance(name, str):
            index = self._actuator_name2id[name]
            self.data.ctrl[index] = self.init_ctrl_ref[name] + val
        elif isinstance(name, list) and isinstance(val, list):
            for n, v in zip(name, val):
                index = self._actuator_name2id[n]
                self.data.ctrl[index] = self.init_ctrl_ref[n] + v

    def set_ctrl(self, name, val):
        # set absolute control signal
        if isinstance(name, str):
            index = self._actuator_name2id[name]
            self.data.ctrl[index] = val
        elif isinstance(name, list) and isinstance(val, list):
            for n, v in zip(name, val):
                index = self._actuator_name2id[n]
                self.data.ctrl[index] = v

    def reset_data_keyframe(self,key_id = 0):
        mujoco.mj_resetDataKeyframe(self.robot.model, self.robot.data, key_id)

    def set_ctrl_from_keyframe(self, key_id = 0):
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        if self.model.joint(0).type == mujoco.mjtJoint.mjJNT_FREE:
            self.data.ctrl = self.data.qpos[7:]
        else:
            self.data.ctrl = self.data.qpos

    def get_body_pos(self, name):
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.body(i).xpos, self.data.body(i).xquat

        # Update the qpos of B with the newly arranged values
        # self.qpos = new_qpos_B
        #self.ctrl[7:] = new_qpos_B[7:]
        # self.step()
    @property
    def qpos(self):
        return self.data.qpos

    def gain_amplify(self, amplitude):
        for i in range(len(self.model.actuator_gainprm)):
            self.model.actuator_gainprm[i, 0] *= amplitude

    def body_id2name(self, id):
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, id)

    def ctrl(self, name, val):
        self.data.ctrl[self.actuators[name]] = val

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_contact_pair(self):
        pair = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = self.body_id2name(self.model.geom_bodyid[con.geom1])
            g2 = self.body_id2name(self.model.geom_bodyid[con.geom2])
            if g1 != "world" and g2 != "world":
                pair.append((g1, g2))
        return pair
    def get_joint_pos(self,names):
        if isinstance(names, str):
            return self.data.qpos[self._joint_name2id[names]]
        elif isinstance(names, list):
            return [self.data.qpos[self._joint_name2id[name]] for name in names]
    def get_joint_vel(self,names):
        if isinstance(names, str):
            return self.data.qvel[self._joint_name2id[names]]
        elif isinstance(names, list):
            return [self.data.qvel[self._joint_name2id[name]] for name in names]

    def print_contact_pair(self):
        """
        prints list of contact pair to be excluded
        """

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1 = self.body_id2name(self.model.geom_bodyid[con.geom1])
            g2 = self.body_id2name(self.model.geom_bodyid[con.geom2])
            if g1 != "world" and g2 != "world":
                print(f"<exclude body1=\"{g1}\" body2=\"{g2}\"/>")


def viewer_handle(robot: MujocoRobot):

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        mujoco.mj_resetDataKeyframe(robot.model, robot.data, 0)
        robot.step()
        # robot.data.ctrl = np.zeros_like(robot.data.qpos[7:])
        if(len(robot.data.ctrl) > 0):
            robot.data.ctrl = robot.data.qpos[7:]
        robot.print_contact_pair()

        wall = time.monotonic()
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0


        while viewer.is_running():
            t = time.monotonic() - wall
            while (robot.data.time <= t):
                robot.step()
            viewer.sync()


def update_meshdir(xml_file_path, relative_path, root_link="root_link", add_floating=True):
    """
    returns urdf with meshdir tag for mujoco
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    mujoco_elem = root.find('mujoco')
    if mujoco_elem is None:
        mujoco_elem = ET.Element('mujoco')

    compiler = mujoco_elem.find('.//compiler[@meshdir]')
    if compiler is None:
        compiler = ET.Element(
            'compiler', {"angle": "radian", "autolimits": "true", 'meshdir': relative_path, 'discardvisual': "false", 'balanceinertia': "true"})
        # root.append(compiler)
        mujoco_elem.insert(0, compiler)
    else:
        compiler.set('meshdir', relative_path)

    if(add_floating):
        root.insert(0, mujoco_elem)
        world_link = ET.Element('link', {'name': 'world'})
        floating_joint = ET.Element(
            'joint', {'name': 'floating_joint', 'type': 'floating'})
        floating_joint.append(ET.Element('parent', {'link': 'world'}))
        floating_joint.append(ET.Element('child', {'link': root_link}))
        root.append(world_link)
        root.append(floating_joint)

    return ET.tostring(root, encoding='unicode')


def add_contact_pair(xml_file_path, pair_list):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    # Find the <contact> element
    contact = root.find('.//contact')
    if contact is None:
        contact = ET.Element('contact')
        root.append(contact)

    for pair in pair_list:
        # Create the <exclude> element with the given attributes
        exclude = ET.Element('exclude', {'body1': pair[0], 'body2': pair[1]})
        # Append the <exclude> element to the <contact> element
        contact.append(exclude)

    tree.write(xml_file_path)
    return ET.tostring(root, encoding='unicode')


def add_actuators(xml_file_path, kp=200, damping=None):
    # Load the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Find the <actuator> tag or create it if it doesn't exist
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.Element("actuator")
        root.append(actuator)

    # Loop through each <joint> tag in the <body>
    for body in root.findall(".//body"):
        for joint in body.findall("joint"):
            joint_type = joint.get("type")

            if joint_type != "free":
                joint_name = joint.get("name")
                joint_range = joint.get("range")
                if damping is not None:
                    joint.set("damping", str(damping))

                # Create a new <position> element
                position = ET.Element("position")
                position.set("name", joint_name)
                position.set("joint", joint_name)
                position.set("ctrlrange", joint_range)
                position.set("kp", str(kp))

                # Append the <position> element to <actuator>
                actuator.append(position)
    return ET.tostring(root, encoding='unicode')


def combine_mjcf(model_mjcf_string, scene_mjcf="scene.xml"):
    scene_root = ET.parse(scene_mjcf).getroot()

    # create element tree from xml string
    model_root = ET.fromstring(model_mjcf_string)

    # os.remove("temp.xml")
    for child in model_root:
        scene_root.append(child)
    return ET.tostring(scene_root, encoding='unicode')

# exclude collision pair


def exclude_collisions(xml_string):
    for i in range(10):
        robot = MujocoRobot(xml_string, from_path=False)
        mujoco.mj_resetDataKeyframe(robot.model, robot.data, 0)
        for i in range(10):
            mujoco.mj_step(robot.model, robot.data)
        mujoco.mj_saveLastXML("temp.xml", robot.model)
        contact_pairs = robot.get_contact_pair()
        xml_string = add_contact_pair("temp.xml", contact_pairs)
        os.remove("temp.xml")
        if(len(contact_pairs) < 2):
            break
    return xml_string

def update_meshdir_xml_str(xml_str, mesh_dir):
    """
    returns xml with meshdir updated tag
    """
    tree = ET.fromstring(xml_str)
    
    compiler = tree.find('.//compiler[@meshdir]')
    compiler.set('meshdir', mesh_dir)

    return ET.tostring(tree, encoding='unicode')

def generate_mjcf_scene(filepath, meshdir,root_link="root_link", kp=100, damping=50, add_floating=True):
    # create urdf xml string with mujoco meshdir tag
    urdf_str = update_meshdir(filepath, meshdir, root_link=root_link,add_floating=add_floating)

    robot = MujocoRobot(urdf_str, from_path=False)

    # save urdf to xml
    mujoco.mj_saveLastXML("temp.xml", robot.model)

    # add actuators
    xml_string = add_actuators("temp.xml", kp=kp, damping=damping)

    # exclude collision pair
    xml_string = exclude_collisions(xml_string)

    # combine scene and robot
    xml_string = combine_mjcf(xml_string, "scene.xml")
    return xml_string


if __name__ == "__main__":
    import threading
    description = ["iCubGazeboV2_6.xml"]
    robots = []
    robots.append(MujocoRobot(description[0], [0, 0, 0]))
    # robots.append(MujocoRobot(urdf[2]))
    threads = []
    for robot in robots:
        t = threading.Thread(target=viewer_handle, args=(robot,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
