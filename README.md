## LSCI-TC 
The source code of Semantic_Consistence _Interaction_with_Calibration_Loss_for_Remote_Sensing_Image-Text_Retrieval.
### Experimental Section Summary

The proposed LSCI-TC framework significantly improves retrieval performance in remote sensing image-text retrieval tasks by introducing cross-modal fine-grained interaction and task-adaptive calibration loss. The experimental section mainly includes the following aspects:

1. **Quantitative Comparison and Visualization**: On the RSITMD dataset, LSCI-TC achieves higher mRecall and lower mECE in image-text retrieval tasks, and demonstrates stronger semantic matching ability in Top-5 retrieval results.
2. **Ablation Studies**:
   - By gradually introducing task calibration loss, soft labels, and the LSCI module, mRecall increases from 44.35% to 59.74%, and mECE is significantly reduced, verifying the effectiveness of each module.
   - Comparisons of different attention mechanisms and feature fusion methods within the LSCI module show that residual connections and multi-head interactions can further improve retrieval performance.
   - Hyperparameters such as the initial gamma value of the task calibration loss, the number of bins, and the soft label weight have a significant impact on performance. The best mRecall is achieved when initgamma=-1.0, bin=15, and soft label weight is 0.6-1.0.
3. **Visualization Analysis**: Using methods such as Grad-CAM, LSCI-TC demonstrates more precise word-level localization, better capturing fine-grained semantic correspondences between images and texts.

