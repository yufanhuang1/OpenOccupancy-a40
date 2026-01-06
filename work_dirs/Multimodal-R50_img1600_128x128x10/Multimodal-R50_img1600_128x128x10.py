point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuscOCCDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(896, 1600),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        sequential=False,
        aligned=True,
        trans_only=False,
        depth_gt_path='./data/depth_gt',
        mmlabnorm=True,
        load_depth=True,
        img_norm_cfg=None),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=dict(
            rot_lim=(0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        input_modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False)),
    dict(
        type='LoadOccupancy',
        to_float32=True,
        use_semantic=True,
        occ_path='./data/nuScenes-Occupancy',
        grid_size=[512, 512, 40],
        use_vel=False,
        unoccupied=0,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        cal_visible=False),
    dict(
        type='OccDefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(896, 1600),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        depth_gt_path='./data/depth_gt',
        sequential=False,
        aligned=True,
        trans_only=False,
        mmlabnorm=True,
        img_norm_cfg=None),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=dict(
            rot_lim=(0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        input_modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        is_train=False),
    dict(
        type='LoadOccupancy',
        to_float32=True,
        use_semantic=True,
        occ_path='./data/nuScenes-Occupancy',
        grid_size=[512, 512, 40],
        use_vel=False,
        unoccupied=0,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        cal_visible=False),
    dict(
        type='OccDefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        with_label=False),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_occ', 'points'],
        meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token'])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=({
        'type':
        'NuscOCCDataset',
        'data_root':
        'data/nuscenes/',
        'occ_root':
        './data/nuScenes-Occupancy',
        'ann_file':
        './data/nuscenes/nuscenes_occ_infos_train.pkl',
        'pipeline': [{
            'type': 'LoadPointsFromFile',
            'coord_type': 'LIDAR',
            'load_dim': 5,
            'use_dim': 5
        }, {
            'type': 'LoadPointsFromMultiSweeps',
            'sweeps_num': 10
        }, {
            'type': 'LoadMultiViewImageFromFiles_BEVDet',
            'is_train': True,
            'data_config': {
                'cams': [
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                'Ncams':
                6,
                'input_size': (896, 1600),
                'src_size': (900, 1600),
                'resize': (-0.06, 0.11),
                'rot': (-5.4, 5.4),
                'flip':
                True,
                'crop_h': (0.0, 0.0),
                'resize_test':
                0.0
            },
            'sequential': False,
            'aligned': True,
            'trans_only': False,
            'depth_gt_path': './data/depth_gt',
            'mmlabnorm': True,
            'load_depth': True,
            'img_norm_cfg': None
        }, {
            'type':
            'LoadAnnotationsBEVDepth',
            'bda_aug_conf': {
                'rot_lim': (0, 0),
                'scale_lim': (0.95, 1.05),
                'flip_dx_ratio': 0.5,
                'flip_dy_ratio': 0.5
            },
            'classes': [
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            'input_modality': {
                'use_lidar': True,
                'use_camera': True,
                'use_radar': False,
                'use_map': False,
                'use_external': False
            }
        }, {
            'type': 'LoadOccupancy',
            'to_float32': True,
            'use_semantic': True,
            'occ_path': './data/nuScenes-Occupancy',
            'grid_size': [512, 512, 40],
            'use_vel': False,
            'unoccupied': 0,
            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            'cal_visible': False
        }, {
            'type':
            'OccDefaultFormatBundle3D',
            'class_names': [
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ]
        }, {
            'type': 'Collect3D',
            'keys': ['img_inputs', 'gt_occ', 'points']
        }],
        'classes': [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        'modality': {
            'use_lidar': True,
            'use_camera': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False
        },
        'test_mode':
        False,
        'use_valid_flag':
        True,
        'occ_size': [512, 512, 40],
        'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        'box_type_3d':
        'LiDAR'
    }, ),
    val=dict(
        type='NuscOCCDataset',
        ann_file='./data/nuscenes/nuscenes_occ_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(896, 1600),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                depth_gt_path='./data/depth_gt',
                sequential=False,
                aligned=True,
                trans_only=False,
                mmlabnorm=True,
                img_norm_cfg=None),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                input_modality=dict(
                    use_lidar=True,
                    use_camera=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False),
                is_train=False),
            dict(
                type='LoadOccupancy',
                to_float32=True,
                use_semantic=True,
                occ_path='./data/nuScenes-Occupancy',
                grid_size=[512, 512, 40],
                use_vel=False,
                unoccupied=0,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                cal_visible=False),
            dict(
                type='OccDefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_occ', 'points'],
                meta_keys=[
                    'pc_range', 'occ_size', 'scene_token', 'lidar_token'
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        occ_root='./data/nuScenes-Occupancy',
        data_root='data/nuscenes/',
        occ_size=[512, 512, 40],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    test=dict(
        type='NuscOCCDataset',
        data_root='data/nuscenes/',
        ann_file='./data/nuscenes/nuscenes_occ_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5),
            dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
            dict(
                type='LoadMultiViewImageFromFiles_BEVDet',
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(896, 1600),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                depth_gt_path='./data/depth_gt',
                sequential=False,
                aligned=True,
                trans_only=False,
                mmlabnorm=True,
                img_norm_cfg=None),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                input_modality=dict(
                    use_lidar=True,
                    use_camera=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False),
                is_train=False),
            dict(
                type='LoadOccupancy',
                to_float32=True,
                use_semantic=True,
                occ_path='./data/nuScenes-Occupancy',
                grid_size=[512, 512, 40],
                use_vel=False,
                unoccupied=0,
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                cal_visible=False),
            dict(
                type='OccDefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'gt_occ', 'points'],
                meta_keys=[
                    'pc_range', 'occ_size', 'scene_token', 'lidar_token'
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        occ_root='./data/nuScenes-Occupancy',
        occ_size=[512, 512, 40],
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5),
        dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
        dict(
            type='LoadMultiViewImageFromFiles_BEVDet',
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(896, 1600),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            depth_gt_path='./data/depth_gt',
            sequential=False,
            aligned=True,
            trans_only=False,
            mmlabnorm=True,
            img_norm_cfg=None),
        dict(
            type='LoadAnnotationsBEVDepth',
            bda_aug_conf=dict(
                rot_lim=(0, 0),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            input_modality=dict(
                use_lidar=True,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            is_train=False),
        dict(
            type='LoadOccupancy',
            to_float32=True,
            use_semantic=True,
            occ_path='./data/nuScenes-Occupancy',
            grid_size=[512, 512, 40],
            use_vel=False,
            unoccupied=0,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            cal_visible=False),
        dict(
            type='OccDefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            with_label=False),
        dict(
            type='Collect3D',
            keys=['img_inputs', 'gt_occ', 'points'],
            meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token'])
    ],
    save_best='SSC_mean',
    rule='greater')
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/Multimodal-R50_img1600_128x128x10'
load_from = None
resume_from = None
workflow = [('train', 1)]
plugin = True
plugin_dir = 'projects/occ_plugin/'
img_norm_cfg = None
occ_path = './data/nuScenes-Occupancy'
depth_gt_path = './data/depth_gt'
train_ann_file = './data/nuscenes/nuscenes_occ_infos_train.pkl'
val_ann_file = './data/nuscenes/nuscenes_occ_infos_val.pkl'
occ_size = [512, 512, 40]
lss_downsample = [4, 4, 4]
voxel_x = 0.2
voxel_y = 0.2
voxel_z = 0.2
voxel_channels = [80, 160, 320, 640]
empty_idx = 0
num_cls = 17
visible_mask = False
cascade_ratio = 4
sample_from_voxel = False
sample_from_img = False
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(896, 1600),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
grid_config = dict(
    xbound=[-51.2, 51.2, 0.8],
    ybound=[-51.2, 51.2, 0.8],
    zbound=[-5.0, 3.0, 0.8],
    dbound=[2.0, 58.0, 0.5])
numC_Trans = 80
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)
model = dict(
    type='OccNet',
    loss_norm=True,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        with_cp=True,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_depth_weight=3.0,
        loss_depth_type='kld',
        grid_config=dict(
            xbound=[-51.2, 51.2, 0.8],
            ybound=[-51.2, 51.2, 0.8],
            zbound=[-5.0, 3.0, 0.8],
            dbound=[2.0, 58.0, 0.5]),
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(896, 1600),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        numC_Trans=80,
        vp_megvii=False),
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=[0.1, 0.1, 0.1],
        max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        input_channel=4,
        base_channel=16,
        out_channel=80,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[1024, 1024, 80]),
    occ_fuser=dict(type='VisFuser', in_channels=80, out_channels=80),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=80,
        block_inplanes=[80, 160, 320, 640],
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=[80, 160, 320, 640],
        out_channels=256,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=4,
        sample_from_voxel=False,
        sample_from_img=False,
        final_occ_size=[512, 512, 40],
        fine_topk=15000,
        empty_idx=0,
        num_level=4,
        in_channels=[256, 256, 256, 256],
        out_channel=17,
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0)),
    empty_idx=0)
bda_aug_conf = dict(
    rot_lim=(0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
test_config = dict(
    type='NuscOCCDataset',
    occ_root='./data/nuScenes-Occupancy',
    data_root='data/nuscenes/',
    ann_file='./data/nuscenes/nuscenes_occ_infos_val.pkl',
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=5,
            use_dim=5),
        dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
        dict(
            type='LoadMultiViewImageFromFiles_BEVDet',
            data_config=dict(
                cams=[
                    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                ],
                Ncams=6,
                input_size=(896, 1600),
                src_size=(900, 1600),
                resize=(-0.06, 0.11),
                rot=(-5.4, 5.4),
                flip=True,
                crop_h=(0.0, 0.0),
                resize_test=0.0),
            depth_gt_path='./data/depth_gt',
            sequential=False,
            aligned=True,
            trans_only=False,
            mmlabnorm=True,
            img_norm_cfg=None),
        dict(
            type='LoadAnnotationsBEVDepth',
            bda_aug_conf=dict(
                rot_lim=(0, 0),
                scale_lim=(0.95, 1.05),
                flip_dx_ratio=0.5,
                flip_dy_ratio=0.5),
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            input_modality=dict(
                use_lidar=True,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            is_train=False),
        dict(
            type='LoadOccupancy',
            to_float32=True,
            use_semantic=True,
            occ_path='./data/nuScenes-Occupancy',
            grid_size=[512, 512, 40],
            use_vel=False,
            unoccupied=0,
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            cal_visible=False),
        dict(
            type='OccDefaultFormatBundle3D',
            class_names=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            with_label=False),
        dict(
            type='Collect3D',
            keys=['img_inputs', 'gt_occ', 'points'],
            meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token'])
    ],
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=True,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    occ_size=[512, 512, 40],
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
train_config = ({
    'type':
    'NuscOCCDataset',
    'data_root':
    'data/nuscenes/',
    'occ_root':
    './data/nuScenes-Occupancy',
    'ann_file':
    './data/nuscenes/nuscenes_occ_infos_train.pkl',
    'pipeline': [{
        'type': 'LoadPointsFromFile',
        'coord_type': 'LIDAR',
        'load_dim': 5,
        'use_dim': 5
    }, {
        'type': 'LoadPointsFromMultiSweeps',
        'sweeps_num': 10
    }, {
        'type': 'LoadMultiViewImageFromFiles_BEVDet',
        'is_train': True,
        'data_config': {
            'cams': [
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            'Ncams':
            6,
            'input_size': (896, 1600),
            'src_size': (900, 1600),
            'resize': (-0.06, 0.11),
            'rot': (-5.4, 5.4),
            'flip':
            True,
            'crop_h': (0.0, 0.0),
            'resize_test':
            0.0
        },
        'sequential': False,
        'aligned': True,
        'trans_only': False,
        'depth_gt_path': './data/depth_gt',
        'mmlabnorm': True,
        'load_depth': True,
        'img_norm_cfg': None
    }, {
        'type':
        'LoadAnnotationsBEVDepth',
        'bda_aug_conf': {
            'rot_lim': (0, 0),
            'scale_lim': (0.95, 1.05),
            'flip_dx_ratio': 0.5,
            'flip_dy_ratio': 0.5
        },
        'classes': [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        'input_modality': {
            'use_lidar': True,
            'use_camera': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False
        }
    }, {
        'type': 'LoadOccupancy',
        'to_float32': True,
        'use_semantic': True,
        'occ_path': './data/nuScenes-Occupancy',
        'grid_size': [512, 512, 40],
        'use_vel': False,
        'unoccupied': 0,
        'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        'cal_visible': False
    }, {
        'type':
        'OccDefaultFormatBundle3D',
        'class_names': [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
    }, {
        'type': 'Collect3D',
        'keys': ['img_inputs', 'gt_occ', 'points']
    }],
    'classes': [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    'modality': {
        'use_lidar': True,
        'use_camera': True,
        'use_radar': False,
        'use_map': False,
        'use_external': False
    },
    'test_mode':
    False,
    'use_valid_flag':
    True,
    'occ_size': [512, 512, 40],
    'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    'box_type_3d':
    'LiDAR'
}, )
optimizer = dict(
    type='AdamW',
    lr=0.0006,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
runner = dict(type='EpochBasedRunner', max_epochs=24)
custom_hooks = [dict(type='OccEfficiencyHook')]
gpu_ids = range(0, 1)
