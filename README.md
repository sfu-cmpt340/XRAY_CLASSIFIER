# SFU CMPT 340 Project Template -- XRAY_CLASSIFIER
This repository is a template for your CMPT 340 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

Add a 1-2 line summary of your project here.

## Important Links

| [Timesheet](https://google.com) | [Slack channel](https://google.com) | [Project report](https://google.com) |
|-----------|---------------|-------------------------|


- Timesheet: Link your timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project/
- Slack channel: Link your private Slack project channel.
- Project report: Link your Overleaf project report document.


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
├── scripts                      ## scripts, if needed
├── docs                         ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate amazing
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/) 





REQUIREMENTS.TXT CONTAINS ALL THE MODULES NEEDED TO RUN THE CODE, SIMPLY EXECUTE: pip install -r requirements.txt

For the data, first download the 3 datasets stored under /RawData from [here](https://www.kaggle.com/datasets/c090ee268c931d0b423485dcc61f82b9befc4039236f33ea6155cb0fa4f127d8). The final folder structure should resemble the below, and then you can run /src/results.ipynb to clean and prepare the data, after which it will be stored under a folder called /preprocessed_data:

📦RawData

 ┣ 📂archive_1

 ┃ ┣ 📂COVID

 ┃ ┃ ┣ 📜COVID.png

 ┃ ┃ ┣ 📜COVID_10.png

 ┃ ┃ ┣  .....

 ┃ ┣ 📂NORMAL

 ┃ ┃ ┣ 📜NORMAL.png

 ┃ ┃ ┣ 📜NORMAL_10.png

 ┃ ┃ ┣  .....

 ┃ ┗ 📂PNEUMONIA

 ┃ ┃ ┣ 📜PNEUMONIA.png

 ┃ ┃ ┣ 📜PNEUMONIA_10.png

 ┃ ┃ ┣  .....

 ┣ 📂archive_2

 ┃ ┗ 📂all_images

 ┃ ┃ ┣ 📂Normal

 ┃ ┃ ┃ ┣ 📜CHNCXR_0001_0.png

 ┃ ┃ ┃ ┣ 📜CHNCXR_0002_0.png

 ┃ ┃ ┃ ┣  .....

 ┃ ┃ ┣ 📂Pneumonia
 
 ┃ ┃ ┃ ┣ 📜BACTERIA-1008087-0001.jpeg

 ┃ ┃ ┃ ┣ 📜BACTERIA-1025587-0001.jpeg

 ┃ ┃ ┃ ┣  .....

 ┃ ┃ ┣ 📂Tuberculosis

 ┃ ┃ ┃ ┣ 📜CHNCXR_0327_1.png

 ┃ ┃ ┃ ┣ 📜CHNCXR_0328_1.png

 ┃ ┃ ┃ ┣  .....

 ┃ ┃ ┗ 📂universal_test

 ┃ ┃ ┃ ┣ 📂Normal

 ┃ ┃ ┃ ┃ ┣ 📜CHNCXR_0153_0.png

 ┃ ┃ ┃ ┃ ┣ 📜CHNCXR_0158_0.png

 ┃ ┃ ┃ ┃ ┣  .....

 ┃ ┃ ┃ ┣ 📂Pneumonia

 ┃ ┃ ┃ ┃ ┣ 📜BACTERIA-1083680-0012.jpeg

 ┃ ┃ ┃ ┃ ┣ 📜BACTERIA-1351146-0002.jpeg

 ┃ ┃ ┃ ┃ ┣  .....

 ┃ ┃ ┃ ┗ 📂Tuberculosis

 ┃ ┃ ┃ ┃ ┣ 📜CHNCXR_0333_1.png

 ┃ ┃ ┃ ┃ ┣ 📜CHNCXR_0365_1.png

 ┃ ┃ ┃ ┃ ┣  .....

 ┗ 📂archive_3

 ┃ ┗ 📂chest_xray

 ┃ ┃ ┣ 📂test

 ┃ ┃ ┃ ┣ 📂NORMAL

 ┃ ┃ ┃ ┃ ┣ 📜IM-0001-0001.jpeg

 ┃ ┃ ┃ ┃ ┣ 📜IM-0003-0001.jpeg

 ┃ ┃ ┃ ┃ ┣  .....

 ┃ ┃ ┃ ┗ 📂PNEUMONIA

 ┃ ┃ ┃ ┃ ┣ 📜person100_bacteria_475.jpeg

 ┃ ┃ ┃ ┃ ┣ 📜person100_bacteria_477.jpeg

 ┃ ┃ ┃ ┃ ┣  .....

 ┃ ┃ ┣ 📂train

 ┃ ┃ ┃ ┣ 📂NORMAL

 ┃ ┃ ┃ ┃ ┣ 📜IM-0115-0001.jpeg

 ┃ ┃ ┃ ┃ ┣ 📜IM-0117-0001.jpeg

 ┃ ┃ ┃ ┃ ┣  .....

 ┃ ┃ ┃ ┗ 📂PNEUMONIA

 ┃ ┃ ┃ ┃ ┣ 📜person1000_bacteria_2931.jpeg

 ┃ ┃ ┃ ┃ ┣ 📜person1000_virus_1681.jpeg

 ┃ ┃ ┃ ┃ ┣  .....

 ┃ ┃ ┗ 📂val

 ┃ ┃ ┃ ┣ 📂NORMAL

 ┃ ┃ ┃ ┃ ┣ 📜NORMAL2-IM-1427-0001.jpeg

 ┃ ┃ ┃ ┃ ┣ 📜NORMAL2-IM-1430-0001.jpeg

 ┃ ┃ ┃ ┃ ┣  .....

 ┃ ┃ ┃ ┗ 📂PNEUMONIA

 ┃ ┃ ┃ ┃ ┣ 📜person1946_bacteria_4874.jpeg

 ┃ ┃ ┃ ┃ ┣ 📜person1946_bacteria_4875.jpeg

 ┃ ┃ ┃ ┃ ┣  .....
 