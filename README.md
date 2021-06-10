# MTML_Pth

Contents of directory scritps:

  1. Training Scripts:
       
       singletask_training.py -  Model Training functions related to training a single task at a time<br>
       multitask_training.py  -  Model Training functions related to training a task and depth (Multi-task) at a time
       
  2. Utility Scripts:
       
       utils.py -  Utility functions including Dataloaders, Augmentations, Transformations, Early Stopping, Performance Metrics, etc.
       
  3. Task Specific Scripts: 

       segmentation_40_13.py - Single task Segmentation script<br>
       Seg_depth.py          - Segmentation + Depth script<br>
       surface_normal.py     - Single task Surface Normal script<br>
       SN_depth.py           - Surface Normal + Depth script<br>
       vanishing_point.py    - Single task Vanishing Point script<br>
       VP_depth.py           - Vanishing Point + Depth script<br>
