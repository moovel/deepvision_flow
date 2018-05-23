
# deepvision_flow
To run flow in inference mode: python run_inference.py path_to_image_folder path_to_pretrained_weight. Find a test folder with sample images and sample output attached.

Weight provided for Kitti street dataset, but specified weights for occlusion/non_occlusion and synthetic Sintel as well as Middleburry will be uploaded to the drive as well. 

Find weight as download here: https://drive.google.com/file/d/1Oz1iC3YN-nwtedM5DxvT6pMvNgwz-gDN/view?usp=sharing

For dataset reference see:

http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow
http://sintel.is.tue.mpg.de/results

The method can be found under deformable flow in the sintel benchmark. Results are NOT fully fine tuned yet and will improve accordingly. http://sintel.is.tue.mpg.de/results . Results are however far superior to the results published in the official flownet 1/2 paper. 






