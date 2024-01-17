# Taming Self-Training for Open-Vocabulary Object Detection

Fork from the Official implementation of online self-training and a split-and-fusion (SAF) head for Open-Vocabulary Object Detection (OVD)

[arXiv](https://arxiv.org/abs/2308.06412)

## Training on COCO-OVD
```bash
python3 ./train_net.py \
        --num-gpus 1 \
        --config-file ./sas_det/configs/ovd_coco_R50_C4_ensemble_PLs.yaml \
        MODEL.WEIGHTS pretrained_ckpt/model_final.pth \
        MODEL.CLIP.OFFLINE_RPN_CONFIG ./sas_det/configs/regionclip/COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \
        MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_coco_48.pth \
        MODEL.CLIP.TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_48_base_cls_emb.pth \
        MODEL.CLIP.CONCEPT_POOL_EMB ./pretrained_ckpt/concept_emb/my_coco_48_base_17_cls_emb.pth \
        MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH ./pretrained_ckpt/concept_emb/coco_65_cls_emb.pth \
        MODEL.ROI_HEADS.SOFT_NMS_ENABLED True \
        MODEL.ENSEMBLE.TEST_CATEGORY_INFO "./datasets/coco_ovd_continue_cat_ids.json" \
        MODEL.ENSEMBLE.ALPHA 0.3 MODEL.ENSEMBLE.BETA 0.7
```

Thanks the author for the open-source code.



