# Traning data generation

First we obtain watertight and simplified meshes via https://github.com/davidstutz/mesh-fusion.

Then we us Pytorch3D to render depth maps and normal maps. 

Finally, we get sketch from depth and normal via opencv's adaptive threshold algorithm.
