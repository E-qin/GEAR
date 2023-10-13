

# ELAM
nohup accelerate launch train.py -c ex/config/ELAM/GO.json  > ex/log/ELAM/GO.log 2>&1 &  # bf16
# LeCaRDv2
nohup accelerate launch train_LeCaRDv2_join.py -c ex/config/LeCaRD_version2/GO.json  > ex/log/LeCaRD_version2/GO.log 2>&1 &  # bf16
