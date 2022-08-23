_base_ = "base.py"

dataset_type = 'VOCDataset'
data_root = 'data/voc/'

data = dict(
    samples_per_gpu=10,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type=dataset_type,
            ann_file=data_root + "VOC2007/ImageSets/Main/trainval.txt",
            img_prefix=data_root + "VOC2007/",
        ),
        unsup=dict(
            type=dataset_type,
            ann_file=data_root + "VOC2012/ImageSets/Main/trainval.txt",
            img_prefix=data_root + "VOC2012/",
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "VOC2007-test/ImageSets/Main/test.txt",
        img_prefix=data_root + 'VOC2007-test/'),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "VOC2007-test/ImageSets/Main/test.txt",
        img_prefix=data_root + 'VOC2007-test/'),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 2],
        )
    ),
)

fold = 1
percent = 10

work_dir = "work_dirs/voc"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="semi-voc",
                name="${cfg_name}",
                config=dict(
                    fold="${fold}",
                    percent="${percent}",
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
