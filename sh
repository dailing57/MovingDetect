python data/preprocess/process_kitti.py  /media/dl/data_pc/data_demo/data_scene_flow /media/dl/data_pc/data_demo_pre/KITTI_processed_occ_final
python run.py -c configs/train/flowstep3d_self_pre.yaml
