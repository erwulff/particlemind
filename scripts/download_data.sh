#!/bin/bash
set -e

mkdir -p data
cd data

wget https://zenodo.org/records/14930758/files/p8_ee_tt_ecm365_rootfiles.tgz?download=1 -O p8_ee_tt_ecm365_rootfiles.tgz
tar xf p8_ee_tt_ecm365_rootfiles.tgz
mkdir -p p8_ee_tt_ecm365/root
mv p8_ee_tt_ecm365_rootfiles/*.root p8_ee_tt_ecm365/root/

rm -f p8_ee_tt_ecm365_rootfiles.tgz
