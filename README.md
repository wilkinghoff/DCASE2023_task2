# DCASE2023_task2

Submission for task 2 "First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring" of the DCASE challenge 2023 ([https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)). The system is an adaptation of the  conceptually simple outlier exposed ASD system specifically designed for domain generalization (https://github.com/wilkinghoff/icassp2023) and uses the sub-cluster AdaCos loss (https://github.com/wilkinghoff/sub-cluster-AdaCos).

The implementation is based on Tensorflow 2.3 (more recent versions can run into problems with the current implementation). Just start the main script for training and evaluation. To run the code, you need to download the development dataset, additional training dataset and the evaluation dataset, and store the files in an './eval_data' and a './dev_data' folder.

When finding this code helpful, or reusing parts of it, a citation would be appreciated:
@techreport{wilkinghoff2023fkie,
  title={Fraunhofer FKIE submission for Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring},
  author={Wilkinghoff, Kevin},
  year={2023},
  institution={DCASE2023 Challenge, Tech. Rep}
}
