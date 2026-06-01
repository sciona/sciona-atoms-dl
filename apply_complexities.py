import json
import glob
import os

all_extracted_path = "/Users/conrad/.gemini/antigravity-cli/brain/e403b54c-33d6-4694-8d85-b8fcae3746f5/scratch/all_extracted_atoms.json"
with open(all_extracted_path, 'r') as f:
    atoms_data = json.load(f)

# Define complex mappings module-by-module
complexity_map = {}

# 1. dl/adversarial/cdg.json
complexity_map.update({
    "auxiliary_logit_loss_fusion": {
        "time": "O(B * C)",
        "space": "O(B * C)",
        "reasoning": "Computes softmax cross-entropy over main and optional auxiliary logits of shape (B, C), scaling linearly with the number of elements.",
        "confidence": 95
    },
    "std_normalized_momentum_gradient": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Normalizes the current gradient and accumulates it into the previous momentum array of size N in linear time and space.",
        "confidence": 95
    },
    "ensemble_prediction_label_inference": {
        "time": "O(M * B * C)",
        "space": "O(B * C)",
        "reasoning": "Sums M model predictions of shape (B, C) and computes the argmax class labels for each batch element.",
        "confidence": 95
    }
})

# 2. dl/architectures/cdg.json
opaque_backbones = ["efficientnet_backbone", "resnet_family_backbone", "densenet_backbone", "swin_transformer_backbone"]
for b in opaque_backbones:
    complexity_map[b] = {
        "time": "O(B * D)",
        "space": "O(B * D)",
        "reasoning": "Allocates a zero-valued feature vector array of shape (B, D) representing the pooled backbone feature boundary.",
        "confidence": 90
    }

complexity_map.update({
    "unet_2d_segmentation": {
        "time": "O(B * C * H * W)",
        "space": "O(B * C * H * W)",
        "reasoning": "Allocates a dense class logits segmentation mask of shape (B, C, H, W) in linear time and space.",
        "confidence": 90
    },
    "unet_1d_sequence": {
        "time": "O(B * C * L)",
        "space": "O(B * C * L)",
        "reasoning": "Allocates sequence logits of shape (B, C, L) where B is batch size, C is classes, and L is sequence length.",
        "confidence": 90
    },
    "yolo_object_detector": {
        "time": "O(B * C * P)",
        "space": "O(B * C * P)",
        "reasoning": "Allocates bounding box detector logits of shape (B, 4 + C, P) where P is the number of raw predictions.",
        "confidence": 90
    },
    "whisper_asr_transformer": {
        "time": "O(B * T * V)",
        "space": "O(B * T * V)",
        "reasoning": "Allocates vocabulary logit tensors of shape (B, T, V) in linear time and space with respect to the output size.",
        "confidence": 90
    },
    "autoregressive_transformer_decoder": {
        "time": "O(B * T * V)",
        "space": "O(B * T * V)",
        "reasoning": "Allocates vocabulary logit tensors of shape (B, T, V) in linear time and space with respect to the output size.",
        "confidence": 90
    },
    "recurrent_sequence_model": {
        "time": "O(B * T * H)",
        "space": "O(B * T * H)",
        "reasoning": "Allocates recurrent hidden states of shape (B, T, H) in linear time and space with respect to the sequence output size.",
        "confidence": 90
    },
    "slowfast_video_network": {
        "time": "O(B * C)",
        "space": "O(B * C)",
        "reasoning": "Allocates class logits of shape (B, C) in linear time and space with respect to the output size.",
        "confidence": 90
    },
    "mil_attention_aggregator": {
        "time": "O(B * C)",
        "space": "O(B * C)",
        "reasoning": "Allocates slide-level class logits of shape (B, C) in linear time and space with respect to the output size.",
        "confidence": 90
    },
    "ctc_decoder_head": {
        "time": "O(B * T * V)",
        "space": "O(B * T)",
        "reasoning": "Decodes sequence alignments from encoder logits of shape (B, T, V) to target token sequences of length T.",
        "confidence": 85
    },
    "protein_embedding_transformer": {
        "time": "O(L * D)",
        "space": "O(D)",
        "reasoning": "Extracts a fixed D-dimensional embedding vector from a protein sequence string of length L.",
        "confidence": 85
    },
    "extreme_multilabel_classifier": {
        "time": "O(L + V)",
        "space": "O(V)",
        "reasoning": "Maps input text of length L to a high-dimensional label score vector of size V in linear time and space.",
        "confidence": 85
    },
    "transformer_encoder_layer": {
        "time": "O(B * T^2 * D)",
        "space": "O(B * T^2)",
        "reasoning": "Computes scaled dot-product self-attention weights of shape (B, H, T, T) over sequence length T.",
        "confidence": 85
    },
    "masked_language_model_pretrain": {
        "time": "O(B * T * D)",
        "space": "O(B * T * D)",
        "reasoning": "Processes tokens of shape (B, T) to predict masked values using a fine-tuned model state.",
        "confidence": 85
    },
    "flex_attention_mask": {
        "time": "O(B * H * T^2)",
        "space": "O(B * H * T^2)",
        "reasoning": "Computes attention weights of shape (B, H, T, T) from custom mask patterns.",
        "confidence": 85
    },
    "graph_transformer_encoder": {
        "time": "O(N * D + E * D_e)",
        "space": "O(N * D + E * D_e)",
        "reasoning": "Updates node representations by passing D-dimensional features across E edges in linear time and space.",
        "confidence": 85
    },
    "soft_attention_alignment": {
        "time": "O(K * D_img)",
        "space": "O(K)",
        "reasoning": "Aligns a text query vector with K image region keys to produce soft attention weights.",
        "confidence": 85
    },
    "multimodal_bilinear_fusion": {
        "time": "O(B * D_1 * D_2)",
        "space": "O(B * D_1 * D_2)",
        "reasoning": "Performs a bilinear matrix multiplication to fuse two distinct feature spaces.",
        "confidence": 85
    },
    "lightgbm_train": {
        "time": "O(N * M * D * T)",
        "space": "O(N * D)",
        "reasoning": "Trains a gradient boosted decision tree ensemble over N samples with D features in linear space.",
        "confidence": 85
    },
    "xgboost_train": {
        "time": "O(N * M * D * T)",
        "space": "O(N * D)",
        "reasoning": "Trains a gradient boosted decision tree ensemble over N samples with D features in linear space.",
        "confidence": 85
    },
    "denoising_autoencoder": {
        "time": "O(B * D)",
        "space": "O(B * D)",
        "reasoning": "Reconstructs noise-corrupted features of shape (B, D) in linear time and space.",
        "confidence": 85
    },
    "stochastic_weight_averaging": {
        "time": "O(M * P)",
        "space": "O(P)",
        "reasoning": "Computes the arithmetic average of model weights across M checkpoints containing P parameters.",
        "confidence": 85
    },
    "multi_task_classification_heads": {
        "time": "O(B * sum(C_i))",
        "space": "O(B * sum(C_i))",
        "reasoning": "Allocates classification logits for multiple parallel tasks in linear time and space.",
        "confidence": 85
    },
    "linear_prediction_head": {
        "time": "O(B * D)",
        "space": "O(B * D)",
        "reasoning": "Performs a linear projection of backbone features to output scores in linear time and space.",
        "confidence": 85
    },
    "dual_head_regression": {
        "time": "O(B)",
        "space": "O(B)",
        "reasoning": "Predicts dual continuous targets from batch features in linear time and space.",
        "confidence": 85
    },
    "face_detector": {
        "time": "O(H * W)",
        "space": "O(H * W)",
        "reasoning": "Detects faces in an image of size H x W, scaling linearly with the image pixel count.",
        "confidence": 85
    },
    "graph_neural_network": {
        "time": "O(L * (N * D + E * D))",
        "space": "O(N * D + E * D)",
        "reasoning": "Performs message passing across E edges for N nodes over L graph layers.",
        "confidence": 85
    },
    "sentence_transformer": {
        "time": "O(L * D)",
        "space": "O(D)",
        "reasoning": "Extracts a fixed-size dense embedding vector from an input text sentence of length L.",
        "confidence": 85
    },
    "heterogeneous_graph_sampler": {
        "time": "O(B * fanout)",
        "space": "O(B * fanout)",
        "reasoning": "Samples local neighborhoods recursively for a mini-batch of seed nodes.",
        "confidence": 85
    },
    "htdemucs_source_separation": {
        "time": "O(T)",
        "space": "O(T)",
        "reasoning": "Separates an audio waveform of length T into source stems in linear time and space.",
        "confidence": 85
    },
    "mosaic_augmentation": {
        "time": "O(H * W * C)",
        "space": "O(H * W * C)",
        "reasoning": "Stitches four images of shape (H, W, 3) into a single mosaic image of shape (2H, 2W, 3) in linear time and space.",
        "confidence": 90
    },
    "foldseek_structural_similarity": {
        "time": "O(L_a * L_b)",
        "space": "O(L_a + L_b)",
        "reasoning": "Performs structural sequence alignment between two protein structures of length L_a and L_b.",
        "confidence": 85
    },
    "spatial_squeeze_excitation": {
        "time": "O(C * H * W)",
        "space": "O(C)",
        "reasoning": "Computes global spatial statistics over feature channels to recalibrate channel-wise weights.",
        "confidence": 85
    },
    "sequence_tagging_trainer": {
        "time": "O(B * T * D)",
        "space": "O(B * T * D)",
        "reasoning": "Trains a token-level tagging classifier over sequence length T and batch size B in linear space.",
        "confidence": 85
    },
    "superpoint_extractor": {
        "time": "O(1)",
        "space": "O(1)",
        "reasoning": "Extracts keypoints and descriptors in constant time and space for the boundary interface.",
        "confidence": 95
    },
    "loftr_dense_matcher": {
        "time": "O(1)",
        "space": "O(1)",
        "reasoning": "Matches dense correspondences in constant time and space for the boundary interface.",
        "confidence": 95
    },
    "superglue_matcher": {
        "time": "O(1)",
        "space": "O(1)",
        "reasoning": "Matches sparse keypoint descriptors in constant time and space for the boundary interface.",
        "confidence": 95
    }
})

# 3. dl/back_translation/cdg.json
complexity_map.update({
    "translate_text": {
        "time": "O(L)",
        "space": "O(L)",
        "reasoning": "Translates an input text string of length L using an opaque translation boundary model.",
        "confidence": 95
    }
})

# 4. dl/detection/cdg.json
complexity_map.update({
    "lung_mask_with_bone_removal": {
        "time": "O(H * W * D)",
        "space": "O(H * W * D)",
        "reasoning": "Applies a threshold mask and computes largest connected components over a 3D volume of shape (H, W, D).",
        "confidence": 95
    },
    "anchor_label_mapping_with_iou_dilation": {
        "time": "O(A * G)",
        "space": "O(A * G)",
        "reasoning": "Computes pairwise intersection-over-union matrix between A anchors and G ground-truth boxes to assign labels.",
        "confidence": 95
    },
    "center_feature_extraction_3d": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Extracts local 3D sub-volume patches centered at N point coordinates from input feature maps.",
        "confidence": 95
    },
    "mtcnn_face_detector": {
        "time": "O(H * W)",
        "space": "O(H * W)",
        "reasoning": "Runs multi-stage cascade face detection over an image of shape (H, W, 3) in linear time and space relative to image size.",
        "confidence": 90
    },
    "margin_expanded_face_crop": {
        "time": "O(N * H * W)",
        "space": "O(N * H * W)",
        "reasoning": "Crops and resizes N face bounding boxes with margin expansion to target resolution (H, W).",
        "confidence": 95
    },
    "face_similarity_align": {
        "time": "O(H * W)",
        "space": "O(H * W)",
        "reasoning": "Applies a similarity transform (affine warp) to align face landmarks to a target image resolution (H, W).",
        "confidence": 95
    },
    "iou_matrix": {
        "time": "O(A * B)",
        "space": "O(A * B)",
        "reasoning": "Computes the intersection-over-union matrix pairwise between two box sets of size A and B in linear time.",
        "confidence": 95
    },
    "giou_matrix": {
        "time": "O(A * B)",
        "space": "O(A * B)",
        "reasoning": "Computes the generalized intersection-over-union matrix pairwise between box sets of size A and B.",
        "confidence": 95
    },
    "nms": {
        "time": "O(N^2)",
        "space": "O(N)",
        "reasoning": "Performs greedy non-maximum suppression by sorting N boxes and iteratively pruning overlapping boxes.",
        "confidence": 95
    },
    "soft_nms": {
        "time": "O(N^2)",
        "space": "O(N)",
        "reasoning": "Performs soft non-maximum suppression by iteratively decaying scores of overlapping boxes.",
        "confidence": 95
    },
    "wbf": {
        "time": "O(N^2)",
        "space": "O(N)",
        "reasoning": "Clusters and averages N bounding boxes from multiple models using weighted box fusion in quadratic time.",
        "confidence": 95
    },
    "wbf_1d": {
        "time": "O(N^2)",
        "space": "O(N)",
        "reasoning": "Clusters and averages N 1D temporal spans from multiple models using weighted fusion in quadratic time.",
        "confidence": 95
    },
    "generate_anchors": {
        "time": "O(H * W * A)",
        "space": "O(H * W * A)",
        "reasoning": "Generates base anchors replicated across a feature map grid of size (H, W) for A aspect ratio/size combinations.",
        "confidence": 95
    },
    "encode_boxes": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Computes relative parameter offsets (deltas) for N anchor and ground-truth box pairs in linear time.",
        "confidence": 95
    },
    "decode_boxes": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Applies parameter offsets (deltas) to N anchors to reconstruct bounding boxes in linear time.",
        "confidence": 95
    },
    "nms_1d": {
        "time": "O(N log N)",
        "space": "O(N)",
        "reasoning": "Finds local peaks in a 1D signal of size N and suppresses neighbors within a fixed distance using sorting.",
        "confidence": 95
    },
    "masks_to_boxes": {
        "time": "O(N * H * W)",
        "space": "O(N)",
        "reasoning": "Finds bounding box coordinates by projecting N binary masks of shape (H, W) onto their spatial axes.",
        "confidence": 95
    },
    "associate_boxes": {
        "time": "O(N^3)",
        "space": "O(N^2)",
        "reasoning": "Solves optimal bipartite matching between two sets of up to N boxes using the Hungarian algorithm.",
        "confidence": 95
    },
    "threshold_detections": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Filters N bounding boxes and scores above a confidence threshold in linear time.",
        "confidence": 95
    },
    "coordinate_aware_3d_unet": {
        "time": "O(B * H * W * D)",
        "space": "O(B * H * W * D)",
        "reasoning": "Runs coordinate-aware 3D UNet segmentation over volume batches of shape (B, H, W, D) in linear time and space.",
        "confidence": 90
    }
})

# 5. dl/embeddings/cdg.json
complexity_map.update({
    "l2_normalize": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Computes the L2 norm and normalizes the input array of size N element-wise in linear time and space.",
        "confidence": 95
    },
    "cosine_similarity_matrix": {
        "time": "O(A * B * D)",
        "space": "O(A * B)",
        "reasoning": "Performs row-wise normalization and a matrix multiplication between embeddings of size (A, D) and (B, D).",
        "confidence": 95
    },
    "alpha_query_expansion": {
        "time": "O(K * D)",
        "space": "O(D)",
        "reasoning": "Computes a weighted sum of K neighbor embeddings of dimension D to produce an expanded query vector.",
        "confidence": 95
    },
    "pca_whiten_reduce": {
        "time": "O(N * D * min(N, D))",
        "space": "O(N * D)",
        "reasoning": "Computes the Singular Value Decomposition (SVD) of centered N x D embeddings to project onto principal components.",
        "confidence": 95
    },
    "embedding_delta": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Computes the element-wise difference between original and transformed embeddings of size N.",
        "confidence": 95
    },
    "build_faiss_flat_ip": {
        "time": "O(Q * R * D)",
        "space": "O(Q * K)",
        "reasoning": "Performs brute-force inner-product search for Q queries against R reference embeddings of dimension D to find top-K.",
        "confidence": 95
    },
    "rerank_by_distance": {
        "time": "O(C * D + C log C)",
        "space": "O(C)",
        "reasoning": "Computes Euclidean distances between query and C candidates of dimension D, then sorts them.",
        "confidence": 95
    }
})

# 6. dl/graph/cdg.json
complexity_map.update({
    "node_degree_bucketing": {
        "time": "O(N + E)",
        "space": "O(N)",
        "reasoning": "Computes node degrees from edge index list of size E and bins N nodes into degree buckets.",
        "confidence": 95
    },
    "feature_clip_standardize": {
        "time": "O(N * D)",
        "space": "O(N * D)",
        "reasoning": "Computes mean, standard deviation, clips, and standardizes N x D node features in linear time and space.",
        "confidence": 95
    },
    "time_budget_estimator": {
        "time": "O(N)",
        "space": "O(1)",
        "reasoning": "Estimates required runtime by checking dynamic step patterns in linear time over N node items.",
        "confidence": 95
    },
    "adjacency_smoothing": {
        "time": "O(N * D + E * D)",
        "space": "O(N * D)",
        "reasoning": "Performs neighborhood feature aggregation by summing D-dimensional features of N nodes across E edges.",
        "confidence": 95
    }
})

# 7. dl/image_augmentation/cdg.json
image_augs = [
    "cutmix_apply", "cutout_apply", "gridmask_apply", "mixup_apply", "flip_apply",
    "random_crop_resize_apply", "affine_transform_centered", "brightness_contrast_apply",
    "hue_saturation_shift", "resize_and_pad_apply"
]
for aug in image_augs:
    complexity_map[aug] = {
        "time": "O(H * W * C)",
        "space": "O(H * W * C)",
        "reasoning": "Applies pixel-level transformation to an image of shape (H, W, C) in linear time and space relative to image pixels.",
        "confidence": 95
    }

complexity_map.update({
    "grayscale_convert_apply": {
        "time": "O(H * W * C)",
        "space": "O(H * W)",
        "reasoning": "Converts an H x W x C color image to a single-channel grayscale image in linear time and space.",
        "confidence": 95
    },
    "ben_graham_retinal_preprocess": {
        "time": "O(H * W * C)",
        "space": "O(H * W * C)",
        "reasoning": "Applies Gaussian blurring and local illumination subtraction to a retinal image in linear time.",
        "confidence": 95
    },
    "tta_geometric_average": {
        "time": "O(T * B * C)",
        "space": "O(B * C)",
        "reasoning": "Computes the geometric average of class predictions of shape (B, C) across T test-time augmentation runs.",
        "confidence": 95
    },
    "ten_crop_batch": {
        "time": "O(B * H * W * C)",
        "space": "O(B * H * W * C)",
        "reasoning": "Generates ten spatial crops of shape (H, W, C) for each image in a batch of size B in linear time and space.",
        "confidence": 95
    },
    "tta_10crop_average": {
        "time": "O(B * C)",
        "space": "O(B * C)",
        "reasoning": "Averages predicted class probabilities across 10-crop augmentations for a batch of size B.",
        "confidence": 95
    },
    "fold_ensemble_average": {
        "time": "O(F * B * C)",
        "space": "O(B * C)",
        "reasoning": "Computes the arithmetic average of class probability predictions across F ensemble folds for batch size B.",
        "confidence": 95
    },
    "normalize_imagenet": {
        "time": "O(B * C * H * W)",
        "space": "O(B * C * H * W)",
        "reasoning": "Normalizes a batch of B images of shape (C, H, W) using ImageNet channel mean and standard deviation.",
        "confidence": 95
    },
    "normalize_per_image": {
        "time": "O(B * C * H * W)",
        "space": "O(B * C * H * W)",
        "reasoning": "Normalizes each image in a batch of size B individually to zero mean and unit variance.",
        "confidence": 95
    },
    "min_max_scale": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Scales an input array of size N element-wise into the [0, 1] range in linear time and space.",
        "confidence": 95
    },
    "compose_augmentations": {
        "time": "O(N * H * W * C)",
        "space": "O(H * W * C)",
        "reasoning": "Applies a chain of N sequential image augmentation functions to an image of shape (H, W, C).",
        "confidence": 95
    }
})

# 8. dl/loss/cdg.json
complexity_map.update({
    "miss_penalty_loss": {
        "time": "O(N)",
        "space": "O(1)",
        "reasoning": "Computes a linear penalty for positive targets predicted below confidence threshold in linear time over N samples.",
        "confidence": 95
    },
    "qwk_loss": {
        "time": "O(B * C + C^2)",
        "space": "O(C^2)",
        "reasoning": "Computes quadratic weighted kappa loss over batch B and classes C using pairwise weight and confusion matrices.",
        "confidence": 95
    },
    "ctc_loss": {
        "time": "O(B * T * V)",
        "space": "O(B * T * V)",
        "reasoning": "Computes CTC loss via forward-backward dynamic programming over sequence length T, batch size B, and vocabulary V.",
        "confidence": 95
    },
    "focal_loss": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Computes binary focal loss element-wise over predictions of size N in linear time and space.",
        "confidence": 95
    },
    "lovasz_softmax_loss": {
        "time": "O(B * C log C)",
        "space": "O(B * C)",
        "reasoning": "Computes the Lovasz extension of the Jaccard index using sorting over C classes for batch size B.",
        "confidence": 95
    },
    "dice_loss": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Computes the intersection and union of predictions and targets of size N in linear time.",
        "confidence": 95
    },
    "crps_score": {
        "time": "O(B * C)",
        "space": "O(1)",
        "reasoning": "Computes discrete Continuous Ranked Probability Score over B samples with C CDF bins.",
        "confidence": 95
    },
    "contrastive_loss": {
        "time": "O(B * D)",
        "space": "O(B)",
        "reasoning": "Computes pairwise contrastive loss over B embedding pairs of dimension D in linear time.",
        "confidence": 95
    },
    "triplet_loss": {
        "time": "O(B * D)",
        "space": "O(B)",
        "reasoning": "Computes margin triplet loss over aligned anchor, positive, and negative vectors of shape (B, D).",
        "confidence": 95
    },
    "label_smoothing_ce": {
        "time": "O(B * C)",
        "space": "O(B * C)",
        "reasoning": "Mixes one-hot labels with uniform mass and computes cross-entropy over batch size B and C classes.",
        "confidence": 95
    },
    "weighted_multitask_loss": {
        "time": "O(T)",
        "space": "O(1)",
        "reasoning": "Computes a weighted sum of T scalar task losses in linear time and constant auxiliary space.",
        "confidence": 95
    },
    "multimodal_nll_loss": {
        "time": "O(B * M * T * D)",
        "space": "O(B * M)",
        "reasoning": "Computes mixture trajectory negative log-likelihood over B batches, M modes, T timesteps, and D coordinates.",
        "confidence": 95
    },
    "weighted_bce_loss": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Computes stable binary cross-entropy element-wise with per-sample weights over arrays of size N.",
        "confidence": 95
    },
    "quantile_spread_to_confidence": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Converts and clips lower/upper quantile spreads of size N in linear time and space.",
        "confidence": 95
    }
})

# 9. dl/perception_3d/cdg.json
complexity_map.update({
    "voxelize_point_cloud": {
        "time": "O(N)",
        "space": "O(V)",
        "reasoning": "Bins N points into a 3D grid of V active voxels, scaling linearly with the number of points.",
        "confidence": 95
    },
    "project_image_to_points": {
        "time": "O(N)",
        "space": "O(N)",
        "reasoning": "Projects N 3D point coordinates into the 2D image plane to sample corresponding color/feature values.",
        "confidence": 95
    },
    "rasterize_bev": {
        "time": "O(N + H * W)",
        "space": "O(H * W)",
        "reasoning": "Rasterizes N 3D points onto a birds-eye-view grid of shape (H, W) in linear time.",
        "confidence": 95
    },
    "ransac_homography": {
        "time": "O(I * N)",
        "space": "O(N)",
        "reasoning": "Robustly estimates homography matrix using RANSAC over N matched points with I iterations.",
        "confidence": 95
    }
})

# 10. dl/recommender/cdg.json
complexity_map.update({
    "co_occurrence_matrix": {
        "time": "O(S * L^2)",
        "space": "O(I_unique^2)",
        "reasoning": "Builds a symmetric co-occurrence matrix by checking all pairwise steps in S sessions of average length L.",
        "confidence": 95
    },
    "cooccurrence_candidates": {
        "time": "O(H * C)",
        "space": "O(I_unique)",
        "reasoning": "Retrieves item candidates by aggregating co-occurrence scores for H user-history items across C candidate classes.",
        "confidence": 95
    },
    "als_user_update": {
        "time": "O(F^3 + F * I_u)",
        "space": "O(F^2)",
        "reasoning": "Solves a ridge-regression step using Cholesky factorization of a F x F matrix for a user with I_u rated items.",
        "confidence": 95
    },
    "als_item_update": {
        "time": "O(F^3 + F * U_i)",
        "space": "O(F^2)",
        "reasoning": "Solves a ridge-regression step using Cholesky factorization of a F x F matrix for an item with U_i rating users.",
        "confidence": 95
    },
    "item_popularity_decay": {
        "time": "O(N)",
        "space": "O(I)",
        "reasoning": "Aggregates and decays item interaction records of size N into a popularity mapping for I items.",
        "confidence": 95
    },
    "session_features": {
        "time": "O(L)",
        "space": "O(L)",
        "reasoning": "Computes length, diversity, duration, and timestamp gaps over a session sequence of length L.",
        "confidence": 95
    },
    "user_item_affinity": {
        "time": "O(H)",
        "space": "O(H)",
        "reasoning": "Summarizes interaction counts and time-decayed recency over a user-item history of size H.",
        "confidence": 95
    },
    "reciprocal_rank_fusion": {
        "time": "O(M * K)",
        "space": "O(U)",
        "reasoning": "Fuses M ranked candidate lists of length K by aggregating reciprocal ranks for U unique candidates.",
        "confidence": 95
    },
    "sampled_softmax_loss": {
        "time": "O(B * N)",
        "space": "O(B * N)",
        "reasoning": "Computes cross-entropy over one positive and N sampled negative scores for a batch of size B.",
        "confidence": 95
    },
    "bpr_max_loss": {
        "time": "O(B * N)",
        "space": "O(B * N)",
        "reasoning": "Computes softmax-weighted BPR-Max pairwise ranking loss over B positive items and N negative samples.",
        "confidence": 95
    },
    "uniform_negative_sampling": {
        "time": "O(B * N)",
        "space": "O(B * N)",
        "reasoning": "Uniformly samples N negative item IDs for each of B batch instances while avoiding positive IDs.",
        "confidence": 95
    },
    "in_batch_negative_sampling": {
        "time": "O(B^2 * D)",
        "space": "O(B^2)",
        "reasoning": "Computes a pairwise similarity matrix of size B x B over batch size B and embedding dimension D to use as negatives.",
        "confidence": 95
    },
    "ranking_moments_extractor": {
        "time": "O(M * I)",
        "space": "O(M * I)",
        "reasoning": "Computes mean, standard deviation, skewness, and kurtosis of ranks for I items across M recommender models.",
        "confidence": 95
    }
})

# 11. dl/segmentation/cdg.json
se_augs = ["morphological_close", "morphological_open", "dilate_mask", "erode_mask"]
for s in se_augs:
    complexity_map[s] = {
        "time": "O(H * W * K^2)",
        "space": "O(H * W)",
        "reasoning": "Applies a morphological operation on a binary mask of shape (H, W) using a K x K structuring element.",
        "confidence": 95
    }

complexity_map.update({
    "fill_holes": {
        "time": "O(H * W)",
        "space": "O(H * W)",
        "reasoning": "Fills enclosed background holes inside foreground components of a binary mask of shape (H, W).",
        "confidence": 95
    },
    "filter_components_by_area": {
        "time": "O(H * W)",
        "space": "O(H * W)",
        "reasoning": "Labels connected components and filters those below a pixel area threshold on a mask of shape (H, W).",
        "confidence": 95
    },
    "dense_crf_2d": {
        "time": "O(I * H * W * C)",
        "space": "O(H * W * C)",
        "reasoning": "Refines class probabilities using a DenseCRF model over H x W pixels and C classes for I iterations.",
        "confidence": 95
    },
    "watershed_instances": {
        "time": "O(H * W log(H * W))",
        "space": "O(H * W)",
        "reasoning": "Computes distance transform and runs watershed flooding over an image of shape (H, W).",
        "confidence": 95
    },
    "mask_to_rle": {
        "time": "O(H * W)",
        "space": "O(R)",
        "reasoning": "Encodes a binary mask of shape (H, W) into column-major run-length pairs containing R runs.",
        "confidence": 95
    },
    "rle_to_mask": {
        "time": "O(H * W)",
        "space": "O(H * W)",
        "reasoning": "Decodes column-major run-length pairs into a binary mask of shape (H, W).",
        "confidence": 95
    },
    "smooth_contour": {
        "time": "O(N^2)",
        "space": "O(N)",
        "reasoning": "Simplifies an ordered contour of N points using the Ramer-Douglas-Peucker algorithm in quadratic worst-case time.",
        "confidence": 95
    },
    "wkt_to_mask": {
        "time": "O(P + H * W)",
        "space": "O(H * W)",
        "reasoning": "Rasterizes a polygon of P vertices into a binary mask of shape (H, W) in linear time.",
        "confidence": 95
    },
    "false_color_composite": {
        "time": "O(H * W * 3)",
        "space": "O(H * W * 3)",
        "reasoning": "Stretches three bands of shape (H, W) and stacks them into an 8-bit RGB false-color image.",
        "confidence": 95
    }
})

# 12. dl/skeletonization/cdg.json
complexity_map.update({
    "skeletonize_2d": {
        "time": "O(I * H * W)",
        "space": "O(H * W)",
        "reasoning": "Applies iterative thinning to reduce a binary mask of shape (H, W) to a skeleton in I passes.",
        "confidence": 95
    },
    "medial_axis_2d": {
        "time": "O(H * W)",
        "space": "O(H * W)",
        "reasoning": "Computes a medial axis skeleton and distance transform over a mask of shape (H, W) in linear time.",
        "confidence": 95
    },
    "skeleton_to_graph": {
        "time": "O(S)",
        "space": "O(S)",
        "reasoning": "Converts a centerline skeleton containing S active pixels into a NetworkX MultiGraph via path tracing.",
        "confidence": 95
    }
})

# 13. dl/tabular/cdg.json
complexity_map.update({
    "entity_embedding_lookup": {
        "time": "O(B * F)",
        "space": "O(B * sum(D_i))",
        "reasoning": "Looks up embedding vectors of dimensions D_i across F fields for a batch of size B and concatenates them.",
        "confidence": 95
    }
})

# 14. dl/text_similarity/cdg.json
complexity_map.update({
    "levenshtein_distance": {
        "time": "O(L_1 * L_2)",
        "space": "O(min(L_1, L_2))",
        "reasoning": "Computes minimum edit distance between two strings of lengths L_1 and L_2 using an optimized DP buffer.",
        "confidence": 95
    },
    "jaro_winkler_similarity": {
        "time": "O(L_1 * L_2)",
        "space": "O(L_1 + L_2)",
        "reasoning": "Computes Jaro-Winkler string similarity by finding matching characters and transpositions between two strings.",
        "confidence": 95
    }
})

# 15. dl/time_series/cdg.json
complexity_map.update({
    "exponential_smoothing_level": {
        "time": "O(T)",
        "space": "O(T)",
        "reasoning": "Recursively updates exponential smoothing levels for a univariate time series of length T.",
        "confidence": 95
    },
    "multiplicative_seasonality_decompose": {
        "time": "O(T)",
        "space": "O(T)",
        "reasoning": "Extracts multiplicative seasonal factors over time series observations of length T.",
        "confidence": 95
    },
    "smyl_loss": {
        "time": "O(T)",
        "space": "O(1)",
        "reasoning": "Computes Smyl hybrid loss (sMAPE and MASE) over forecast horizon T in linear time.",
        "confidence": 95
    },
    "pinball_loss": {
        "time": "O(T)",
        "space": "O(1)",
        "reasoning": "Computes quantile pinball regression loss over forecasting steps of length T in linear time.",
        "confidence": 95
    }
})

# 16. dl/training/cdg.json
complexity_map.update({
    "online_hard_negative_mining": {
        "time": "O(N log N)",
        "space": "O(N)",
        "reasoning": "Sorts N negative scores to select the highest-loss elements for hard negative mining.",
        "confidence": 95
    },
    "size_aware_nodule_oversampling": {
        "time": "O(N log N)",
        "space": "O(N)",
        "reasoning": "Sorts N nodule candidates by diameter to rebalance training distributions via size-proportionate sampling.",
        "confidence": 95
    },
    "softmax_temperature_proposal_sampling": {
        "time": "O(N log N)",
        "space": "O(N)",
        "reasoning": "Applies temperature scaling, softmax, and cumulative sampling without replacement over N elements.",
        "confidence": 95
    },
    "ternary_search_threshold": {
        "time": "O(I * N)",
        "space": "O(1)",
        "reasoning": "Optimizes a classification threshold by running ternary search for I iterations over N score samples.",
        "confidence": 95
    },
    "multisample_dropout": {
        "time": "O(M * B * D)",
        "space": "O(B * D)",
        "reasoning": "Applies M independent dropout masks to a batch of shape (B, D) and computes their averaged output.",
        "confidence": 95
    }
})

# 17. dl/video_temporal/cdg.json
complexity_map.update({
    "sample_frame_indices": {
        "time": "O(F_target)",
        "space": "O(F_target)",
        "reasoning": "Computes frame indices approximating target FPS extraction in linear time relative to target frame count.",
        "confidence": 95
    },
    "uniform_sample_indices": {
        "time": "O(K)",
        "space": "O(K)",
        "reasoning": "Selects K uniformly spaced frame indices over the total frame timeline in linear time.",
        "confidence": 95
    },
    "temporal_mean_pool": {
        "time": "O(T * D)",
        "space": "O(D)",
        "reasoning": "Computes the mean activation over the temporal dimension of a sequence of shape (T, D).",
        "confidence": 95
    },
    "temporal_max_pool": {
        "time": "O(T * D)",
        "space": "O(D)",
        "reasoning": "Computes the maximum activation over the temporal dimension of a sequence of shape (T, D).",
        "confidence": 95
    },
    "temporal_attention_pool": {
        "time": "O(T^2 + T * D)",
        "space": "O(T^2 + T * D)",
        "reasoning": "Computes self-attention dot-products of shape (T, T) over T temporal frames with dimension D.",
        "confidence": 95
    },
    "temporal_median_filter": {
        "time": "O(T * K)",
        "space": "O(T)",
        "reasoning": "Applies a local 1D median filter of width K to a sequence of length T in linear time.",
        "confidence": 95
    },
    "sliding_windows": {
        "time": "O(W * S)",
        "space": "O(W * S)",
        "reasoning": "Extracts overlapping temporal windows to output a matrix of shape (W, S) for W windows of size S.",
        "confidence": 95
    },
    "stack_adjacent_frames": {
        "time": "O(H * W * C)",
        "space": "O(H * W * C)",
        "reasoning": "Stacks neighboring grayscale frames into C channels of shape (H, W) to provide temporal context.",
        "confidence": 95
    },
    "temporal_unroll": {
        "time": "O(T * D)",
        "space": "O(T * D)",
        "reasoning": "Repeats block-level predictions of shape (T_block, D) to retrieve original timestep shape (T, D).",
        "confidence": 95
    }
})


# Now update the files!
files_to_update = glob.glob("/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms/**/cdg.json", recursive=True)

print(f"Checking updates for {len(files_to_update)} cdg.json files...")

stats_updated = 0
stats_nodes = 0

for file_path in sorted(files_to_update):
    rel_path = os.path.relpath(file_path, "/Users/conrad/personal/sciona-atoms-dl/src/sciona/atoms")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    updated = False
    
    for node in nodes:
        status = node.get("status")
        if status == "decomposed":
            continue
            
        node_id = node.get("node_id")
        stats_nodes += 1
        
        if node_id in complexity_map:
            cm = complexity_map[node_id]
            node["time_complexity"] = cm["time"]
            node["space_complexity"] = cm["space"]
            node["complexity_reasoning"] = cm["reasoning"]
            node["complexity_confidence"] = cm["confidence"]
            updated = True
        else:
            print(f"Warning: {node_id} in {rel_path} has no complexity mapping defined!")
            
    if updated:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        # Verify
        with open(file_path, 'r') as f:
            verified_data = json.load(f)
            
        stats_updated += 1

print(f"Finished. Updated {stats_updated} JSON files. Processed {stats_nodes} total nodes.")
