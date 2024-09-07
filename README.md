# CSV-Filter: A Comprehensive Structural Variation Filtering Tool for Single Molecule Real-Time Sequencing

## Introduction

Structure Variations (SVs) play an important role in genetic research and precision medicine. However, existing SV detection methods usually contain a substantial number of false positive calls. It is necessary to develop effective filtering approaches. We developed a novel deep learning-based SV filtering tool, CSV-Filter, for both second and third generation sequencing data.
In CSV-Filter, we proposed a novel multi-level grayscale image encoding method based on CIGAR strings of the alignment results and employed image augmentation techniques to improve the extraction of SV features. We also utilized self-supervised learning networks for transfer as classification models, and employed mixed-precision operations to accelerate the training process.
The experimental results show that the integration of CSV-Filter with popular second-generation and third-generation SV detection tools could considerably reduce false positive SVs, while maintaining true positive SVs almost unchanged.
Compared with DeepSVFilter, a SV filtering tool for second-generation sequencing, CSV-Filter can recognize more false positive SVs and supports third-generation sequencing data as an additional feature. 

## Requirements

CSV-Filter is tested to work under:

* Ubuntu 18.04
* gcc 7.5.0
* CUDA 11.3
* cudnn 8200
* nvidia-smi 470.182.03
* Anoconda 4.10.3
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

## Quick start

### Datasets

#### Reference

* GRCh37: [https://ftp.ensembl.org/pub/release-75/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz](https://ftp.ensembl.org/pub/release-75/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz)
* hs37d5: [https://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz](https://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz)

#### HG002

* Tier1 benchmark SV callset and high-confidence HG002 region: [https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/analysis/NIST_SVs_Integration_v0.6/](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/analysis/NIST_SVs_Integration_v0.6/)
* PacBio 70x (CLR): [https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/)
* PacBio CCS 15kb\_20kb chemistry2 (HiFi): [https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_CCS_15kb_20kb_chemistry2/reads/](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_CCS_15kb_20kb_chemistry2/reads/)
* Oxford Nanopore ultralong (guppy-V3.2.4\_2020-01-22): [ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/Ultralong_OxfordNanopore/guppy-V3.2.4_2020-01-22/HG002_ONT-UL_GIAB_20200122.fastq.gz](ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/Ultralong_OxfordNanopore/guppy-V3.2.4_2020-01-22/HG002_ONT-UL_GIAB_20200122.fastq.gz)
* NHGRI_Illumina300X_AJtrio_novoalign_bams: [https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG002.hs37d5.60x.1.bam](https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/NIST_HiSeq_HG002_Homogeneity-10953946/NHGRI_Illumina300X_AJtrio_novoalign_bams/HG002.hs37d5.60x.1.bam)

#### NA12878

* NA12878: [Index of /giab/ftp/data/NA12878/NA12878_PacBio_MtSinai (nih.gov)](https://ftp.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/NA12878_PacBio_MtSinai/)

data used:
* NA12878.sorted.vcf.gz
* NA12878.sorted.vcf.gz.tbi
* sorted_final_merged.bam
* sorted_final_merged.bam.bai
* corrected_reads_gt4kb.fasta 

### Model that have been trained

Download trained models from [Releases · xzyschumacher/CSV-Filter (github.com)](https://github.com/xzyschumacher/CSV-Filter/releases)

### Environment by using anaconda and pip

#### Install Ubuntu support environment
apt: 
```bash
sudo apt-get update
```

gcc: 
```bash
sudo apt-get install gcc
```

g++: 
```bash
sudo apt-get install g++
```

vim: 
```bash
sudo apt-get install vim
```

#### Install drivers
show GPU version and recommended drivers: 
```bash
ubuntu-drivers devices
```

install recommended drivers:
```bash
sudo apt install nvidia-470
reboot
```

test: 
```bash
nvidia-smi
```

#### Install CUDA
CUDA Toolkit Archive：https://developer.nvidia.com/cuda-toolkit-archive

download CUDA Toolkit 11.3.1: 
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda_11.3.1_465.19.01_linux.run
```

run: 
```bash
sudo sh cuda_11.3.1_465.19.01_linux.run
accept
Continue
Install
```

Modify environment variables:
```python
export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
refresh environment variables:
```bash
source ~/.bashrc
```

test: 
```bash
nvcc -V
```

run test program:
```bash
cd ~/NVIDIA_CUDA-11.3_Samples/1_Utilities/bandwidthTest
make
./bandwidthTest
```

#### Install Anaconda
download Anaconda:
```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2021.11-Linux-x86_64.sh
```

run:
```bash
bash Anaconda3-2021.11-Linux-x86_64.sh
yes
```

**default direction: /home/username/anaconda3**

Modify environment variables:
```python
export PATH=”/home/username/anaconda3/bin:$PATH”
```

refresh environment variables: 
```bash
source ~/.bashrc`
```

test: 
```bash
conda --version`
```

#### Create CSV-Filter environment

create a new environment: 
```bash
conda create -n MSVF python=3.6 -y
```

activate CSV-Filter environment: 
```bash
conda activate csv-filter
```

install pytorch/cudatoolkit/torchvision/torchaudio/pytorch: 
```bash
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit==11.3.1 -c pytorch -y
```

install pytorch-lightning: 
```bash
conda install pytorch-lightning=1.5.10 -c conda-forge -y
```

install ray:
```bash
pip install ray[tune]==1.6.0
```

install redis:
```bash
conda install redis -y
```

install scikit-learn:
```bash
conda install scikit-learn -y
```

install matplotlib:
```bash
conda install matplotlib -y
```

install pudb:
```bash
pip install pudb
```

install samtools:
```bash
conda install samtools -c bioconda
```

install hyperopt:
```bash
pip install hyperopt
```

install pysam:
```bash
pip install pysam==0.15.4
```

install parallel:
```bash
sudo apt install parallel
```

## Usage

### Simple train

```
python simple_train.py selected_model
(python simple_train.py resnet50)
```

### Train

`$ cd data`

get BAM file:

`$ wget https://ftp.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/NA12878_PacBio_MtSinai/sorted_final_merged.bam`

parallel index BAM file：

`$ parallel samtools index ::: *.bam`

get VCF file:

`$ wget https://ftp.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/NA12878_PacBio_MtSinai/NA12878.sorted.vcf.gz`

uncompress VCF gz file:

`$ tar -xzvf NA12878.sorted.vcf.gz`

`$ cd ../src`

vcf data preprocess:

`$ python vcf_data_process.py`

BAM data preprocess:

`$ python bam2depth.py`

parallel generate images:
```
$ python parallel_process_file.py --thread_num thread_num  
($ python parallel_process_file.py --thread_num 16)
```

check generated images:

`$ python process_file_check.py`

rearrange generated images:

`$ python data_spread.py`

train:

`$ python train.py`

### Predict & Filter

predict:
```
$ python predict.py selected_model
(e.g. $ python predict.py resnet50)
```

filter:
```
$ python filter.py selected_model
(e.g. $ python filter.py resnet50)
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
