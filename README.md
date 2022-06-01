# TraceRCA
Practical Root Cause Localization for Microservice Systems via Trace Analysis. IWQoS 2021


## Dataset
The study data is public at 
  - OneDrive: https://1drv.ms/u/s!Ao2DxaN2zku_bAUszKmCUiodw94?e=7ThI47
  - Tsinghua Cloud https://cloud.tsinghua.edu.cn/d/8371855eddd64a8db23b/ (中国大陆可访问)


## Implementation Code

The experiment workflow is controlled via the Makefile. The input and output of each step can be referred to the Makefile

- `run_selecting_features.py`: Feature selection
- `run_anomaly_detection_invo.py`: Anomaly detection based on the useful features
- `run_localization_association_rule_mining_20210516.py`: Root-cause service ocalization
- `prepare_train_file_tmp.py` is used to split the dataset into train and test datasets. Note that this step is not included in the Makefile.


[Presentation Video](https://www.bilibili.com/video/BV14b4y1C7rQ/)
## Cite
If the dataset is helpful, please cite the paper.
``` bibtex
@inproceedings{li2021practical,
  title={Practical Root Cause Localization for Microservice Systems via Trace Analysis},
  author={Li, Zeyan and Chen, Junjie and Jiao, Rui and Zhao, Nengwen and Wang, Zhijun and Zhang, Shuwei and Wu, Yanjun and Jiang, Long and Yan, Leiqin and Wang, Zikai and others},
  booktitle={IEEE/ACM International Symposium on Quality of Service (IWQoS) 2021},
  year={2021},
  publisher = {{IEEE}}
}
```
