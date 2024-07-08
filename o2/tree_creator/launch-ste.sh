o2-analysis-track-propagation -b --configuration json://configuration-ste.json |\
 o2-analysis-lf-lambdakzerobuilder -b --configuration json://configuration-ste.json |\
 o2-analysis-timestamp -b --configuration json://configuration-ste.json |\
 o2-analysis-pid-tpc -b --configuration json://configuration-ste.json |\
 o2-analysis-pid-tof-full -b --configuration json://configuration-ste.json |\
 o2-analysis-pid-tof-base -b --configuration json://configuration-ste.json |\
 o2-analysis-pid-tof  -b --configuration json://configuration-ste.json |\
 o2-analysis-pid-tpc-base -b --configuration json://configuration-ste.json |\
 o2-analysis-ft0-corrected-table -b --configuration json://configuration-ste.json |\
 o2-analysis-event-selection -b --configuration json://configuration-ste.json |\
 o2-analysis-multiplicity-table -b --configuration json://configuration-ste.json |\
 o2-analysis-lf-cluster-studies-tree-creator -b --configuration json://configuration-ste.json --aod-file @input_data.txt --aod-writer-keep dangling
 #o2-analysis-bc-converter -b --configuration json://configuration-ste.json |\
 #o2-analysis-tracks-extra-converter -b --configuration json://configuration-ste.json |\
 #o2-analysis-v0converter -b --configuration json://configuration-ste.json 