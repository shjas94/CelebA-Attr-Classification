# CelebA Classification (Smiling, Wavy Hair, Male)

# Code Structure
```
.
├── README.md
├── checkpoint
├── data
│   ├── img_align_celeba.zip
│   ├── list_attr_celeba.csv.zip
│   └── list_eval_partition.csv.zip
├── dataset
│   ├── augmentations.py
│   └── dataset.py
├── inference.py
├── main.py
├── models
│   ├── models.py
│   └── modules.py
├── trainer.py
└── utils
    ├── earlystopping.py
    ├── scheduler.py
    └── utils.py
```

# How to Train
```
$ python main.py --run_name="Your run name"
```

# How to Evaluate
```
$ python inference.py --model_name="Your model_name"
```