#!bin/bash
YELLOW='\033[1;33m'
NC='\033[0m'

mkdir ../data
cd ../data

# Training Set
echo -e "${YELLOW} Downloading Training Data"
echo -e "${NC}"
wget https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip
wget https://data.broadinstitute.org/bbbc/BBBC038/stage1_train_labels.csv

echo -e "${YELLOW} Downloading Test Data part 1"
echo -e "${NC}"
wget https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip
wget https://data.broadinstitute.org/bbbc/BBBC038/stage1_solution.csv

echo -e "${YELLOW} Downloading Test Data part 2"
echo -e "${NC}"
wget https://data.broadinstitute.org/bbbc/BBBC038/stage2_test_final.zip
wget https://data.broadinstitute.org/bbbc/BBBC038/stage2_solution_final.csv