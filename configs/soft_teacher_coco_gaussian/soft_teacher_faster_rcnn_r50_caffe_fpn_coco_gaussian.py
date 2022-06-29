_base_ = "base.py"

data_root = 'data/coco/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="data/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="data/coco/train2017/",
        ),
    ),
    val=dict(
        type="CocoDataset",
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'),
    test=dict(
        type="CocoDataset",
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

fold = 1
percent = 10

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="GaussianTeacher",
        #         name="COCO_Gaussian_FRCNN_WO_JITTER",
        #         config=dict(
        #             fold="${fold}",
        #             percent="${percent}",
        #             work_dirs="${work_dir}",
        #             total_step="${runner.max_iters}",
        #         ),
        #     ),
        #     by_epoch=False,
        # ),
    ],
)
