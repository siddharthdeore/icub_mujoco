# icub mujoco

Some good pyhon snipets to generate Mujoco MJCF files for [ergoCub](https://github.com/icub-tech-iit/ergocub-software) and [iCub](https://github.com/robotology/icub-models.git) descriptions. goal is to keep it compatible with original *cub repos.

Supported Models:
```
iCubGazeboV2_6, iCubGazeboV2_5, iCubGazeboV2_7, iCubGenova09
ergoCubGazeboV1, ergoCubGazeboV1_1, ergoCubSN000, ergoCubSN001
```

Unsupported:

following models do not load out of the box; Mujoco does not support DAE mesh. Please replace or convert the DAE mesh file to STL for compatibility.
```
iCubGenova03, iCubParis01, iCubLisboa01, iCubParis02, iCubNancy01
ergoCubGazeboV1_1_minContacts ergoCubGazeboV1_minContacts
```
# Requrements
```
pip install mujoco
```

# Usage
Download the repository and submodules
```
git clone https://github.com/siddharthdeore/icub_mujoco.git --recursive
cd icub_mujoco
```
Generate model `$python generate.py <model_name>`
```
python generate.py iCubGazeboV2_6
```

# Test
```
python example.py
```


# Maintainers
This repository is maintained by:

| <img src="https://avatars.githubusercontent.com/u/12745747" width="32">  | [Siddharth Deore](https://github.com/siddharthdeore) |
|--|--|