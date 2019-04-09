[![pipeline status](https://gitlab.com/VladyslavUsenko/basalt-headers/badges/master/pipeline.svg)](https://gitlab.com/VladyslavUsenko/basalt-headers/commits/master)
[![coverage report](https://gitlab.com/VladyslavUsenko/basalt-headers/badges/master/coverage.svg)](https://gitlab.com/VladyslavUsenko/basalt-headers/commits/master)

## Basalt Headers
This repository contains reusable components of Basalt project as header-only library.

This library includes:
* Camera models.
* Uniform B-Splines for Rd (d-dimentional vectors), SO(3) and SE(3).
* Preintegration of inertial-measurement unit (IMU) measurements.
* Data types to store IMU-camera calibration.


## Related Publications
Camera models implemented in this project are described here: [arXiv:1807.08957](https://arxiv.org/abs/1807.08957)
```
@inproceedings{usenko3dv18, 
    author={V. Usenko and N. Demmel and D. Cremers}, 
    booktitle={2018 International Conference on 3D Vision (3DV)}, 
    title={The Double Sphere Camera Model}, 
    year={2018},
    pages={552-560}, 
    doi={10.1109/3DV.2018.00069}, 
    ISSN={2475-7888}, 
    month={Sep.},
    epub={https://arxiv.org/abs/1807.08957}
}
```

## Licence

The code for this practical course is provided under a BSD 3-clause license. See the LICENSE file for details.
Note also the different licenses of thirdparty submodules.