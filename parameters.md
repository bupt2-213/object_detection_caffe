1.train my rpn
```bash
./tools/train_my_rpn.py --gpu 0 --weight data/imagenet_models/VGG16.v2.caffemodel --imdb voc_2007_trainval --cfg experiments/cfgs/faster_rcnn_alt_opt.yml --set EXP_DIR foobar
```

