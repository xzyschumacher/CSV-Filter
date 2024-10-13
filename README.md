# CSV-Filter: A Comprehensive Structural Variation Filtering Tool for Single Molecule Real-Time Sequencing

## Introduction

Structure Variations (SVs) play an important role in genetic research and precision medicine. However, existing SV detection methods usually contain a substantial number of false positive calls. It is necessary to develop effective filtering approaches. We developed a novel deep learning-based SV filtering tool, CSV-Filter, for both second and third generation sequencing data.
In CSV-Filter, we proposed a novel multi-level grayscale image encoding method based on CIGAR strings of the alignment results and employed image augmentation techniques to improve the extraction of SV features. We also utilized self-supervised learning networks for transfer as classification models, and employed mixed-precision operations to accelerate the training process.
The experimental results show that the integration of CSV-Filter with popular second-generation and third-generation SV detection tools could considerably reduce false positive SVs, while maintaining true positive SVs almost unchanged.
Compared with DeepSVFilter, a SV filtering tool for second-generation sequencing, CSV-Filter can recognize more false positive SVs and supports third-generation sequencing data as an additional feature. 

## Installation

```bash
conda env create -f environment.yml
conda activate csv-filter
```

## Dependence

CSV-Filter is tested to work under:

* Python 3.6
* pysam 0.15.4
* pytorch 1.10.2 
* pytorch-lightning 1.5.10
* hyperopt 0.2.7 
* matplotlib 3.3.4
* numpy 1.19.2
* pudb 2022.1.3
* redis 4.3.6
* samtools 1.5
* scikit-learn 0.24.2
* torchvision 1.10.2
* tensroboard 2.11.2 

## Datasets

### Reference

* GRCh37: [https://ftp.ensembl.org/pub/release-75/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz](https://ftp.ensembl.org/pub/release-75/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz)
* hs37d5: [https://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz](https://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz)

### HG002

* Tier1 benchmark SV callset and high-confidence HG002 region: [https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/analysis/NIST_SVs_Integration_v0.6/](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/analysis/NIST_SVs_Integration_v0.6/)
* PacBio 70x (CLR): [https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/)
* PacBio CCS 15kb\_20kb chemistry2 (HiFi): [https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_CCS_15kb_20kb_chemistry2/reads/](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_CCS_15kb_20kb_chemistry2/reads/)
* Oxford Nanopore ultralong (guppy-V3.2.4\_2020-01-22): [ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/Ultralong_OxfordNanopore/guppy-V3.2.4_2020-01-22/HG002_ONT-UL_GIAB_20200122.fastq.gz](ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/Ultralong_OxfordNanopore/guppy-V3.2.4_2020-01-22/HG002_ONT-UL_GIAB_20200122.fastq.gz)
* NHGRI_Illumina300X_AJtrio_novoalign_bams: [https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG002.hs37d5.60x.1.bam](https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG002.hs37d5.60x.1.bam)

### NA12878

* NA12878: [Index of /giab/ftp/data/NA12878/NA12878_PacBio_MtSinai (nih.gov)](https://ftp.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/NA12878_PacBio_MtSinai/)

## Model that have been trained

Download trained models from [Releases · xzyschumacher/CSV-Filter (github.com)](https://github.com/xzyschumacher/CSV-Filter/releases)

## Usage

### Train

In the src file

vcf data preprocess:
```bash
python vcf_data_process.py
```

BAM data preprocess:
```bash
python bam2depth.py
```

parallel generate images:
```
python parallel_process_file.py --thread_num thread_num  
(python parallel_process_file.py --thread_num 16)
```

check generated images:
```bash
python process_file_check.py
```

rearrange generated images:
```bash
python data_spread.py
```

train:
```bash
python train.py
```

### Predict & Filter

predict:
```
python predict.py selected_model
(e.g. python predict.py resnet50)
```

filter:
```
python filter.py selected_model
(e.g. python filter.py resnet50)
```

### Switch model to train

In train.py file, modify the data dirction and name of the model.
```python
data_dir = "../data/"
bs = 128
my_label = "resnet50"
```

In net.py file, modify models need to be trianed.
```python
# load local models
self.resnet_model = torch.load("../models/init_resnet50.pt")
self.resnet_model.eval()

# load models from websites
self.resnet_model = torchvision.models.mobilenet_v2(pretrained=True)
self.resnet_model = torchvision.models.resnet34(pretrained=True)
self.resnet_model = torchvision.models.resnet50(pretrained=True)

# load VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised Learning
self.resnet_model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
self.resnet_model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50x2')
self.resnet_model = torch.hub.load('facebookresearch/vicreg:main', 'resnet200x2')
```

PS: sometimes the output dimension is different, so we need to modify the softmax void:
```python
self.softmax = nn.Sequential(
    # nn.Linear(full_dim[-1], 3),
    # nn.Linear(4096, 3),
    # nn.Linear(2048, 3),
    nn.Linear(1000, 3),
    nn.Softmax(1)
)
```

## Citation

Feel free to read and cite our paper in Bioinformatics: [https://academic.oup.com/bioinformatics/article/40/9/btae539/7750355](https://academic.oup.com/bioinformatics/article/40/9/btae539/7750355).

## Contact

For advising, bug reporting and requiring help, please contact [yingbocui@nudt.edu.cn](yingbocui@nudt.edu.cn).
