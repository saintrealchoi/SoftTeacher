_base_ = "base.py"

classes = ('pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor')
data_root = 'data/VisDrone/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="CocoDataset",
            classes = classes,
            ann_file="data/VisDrone/annotations/semi_supervised/instances_train2021.${fold}@${percent}.json",
            img_prefix="data/VisDrone/VisDrone2019-DET-train/",
        ),
        unsup=dict(
            type="CocoDataset",
            classes = classes,
            ann_file="data/VisDrone/annotations/semi_supervised/instances_train2021.${fold}@${percent}-unlabeled.json",
            img_prefix="data/VisDrone/VisDrone2019-DET-train/",
        ),
    ),
    val=dict(
        type="CocoDataset",
        classes=classes,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/'),
    test=dict(
        type="CocoDataset",
        classes=classes,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/'),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
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