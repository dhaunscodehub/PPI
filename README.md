# Prediction of Protein–Protein Interactions Based on Integrating Deep Learning and Feature Fusion



**Clone this repo:**
```commandline
git lfs clone https://gitlab.com/nhanth/DF-PPI.git
```

**Install environments:**
>Tested on python 3.7
```commandline
pip install tensorflow-gpu==2.6.0
pip install keras==2.6.0
pip install gensim==4.3.2
pip install sentencepiece
```

**The programs were tested on the environments:**
1. [ ] pandas==1.3.4
2. [ ] numpy==1.19.5
3. [ ] scikit-learn>=1.0.2
4. [ ] scipy==1.9.3
5. [ ] matplotlib==3.5.1
6. [ ] keras==2.6.0
7. [ ] tensorflow-gpu==2.6.0
8. [ ] gensim==4.3.2
9. [ ] sentencepiece>=0.1.99

## Introduction

1. Folder `data` stores all datasets.
2. Folder `models` stores all defination of models.
3. Folder `protein_descriptors` stores all programs for handcrafted feature extraction.
4. Folder `protein_embedding` stores all programs for protein sequence embedding.
5. Folder `results` stores all obtained results throught experiments.
6. Folder `utils` stores all utilizes programs such as plot, read results,...
7. The following programs are used to perform experiments:
   * For experiment on Human dataset: `experiment_Human.py`
   * For experiment on Yeast core dataset: `experiment_Yeastcore.py`
   * For experiment on Test datasets, PPI network and Cross-species: `experiment_TEST_sets.py`
   * And orther programs.

