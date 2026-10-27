# GMR core and Unitree G1 assets

Source: https://github.com/YanjieZe/GMR
Revision: `25f56db79ac411258b377b6d4ccba0c8ff3372ac`

`motion_retarget.py`, `params.py` and G1 model/meshes are unmodified.
Only meshes referenced by g1_mocap_29dof.xml are included.
The package initializer is replaced with a minimal initializer to avoid unused
SMPL-X, mocap and viewer dependencies. The IK configuration for H36M is our adapter.
GMR license: MIT; robot asset license: assets/unitree_g1/LICENSE.
