
#!/bin/bash
#https://rosettacommons.github.io/foundry/models/rfd3/examples/na_binder_design.html rfd3设计参考网址
#注意当你使用wsl2或者wsl运行时GCC会有问题，export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH，重新指定libstdc++.so.6路径可能解决问题
conda activate RFDF3
rfd3 design out_dir=/your/output/dir inputs=/your/json/document ckpt_path=/path/to/your/checkpoint/files/rfd3_latest.ckpt #详情参考网址 
gunzip  /your/output/dir/*.gz
python3 step_2_MPNN.py -i /home/dupuquan/RFDF3/rfd3_ppi_tutorial/output_cif -o /home/dupuquan/RFDF3/validation_RFDF3/test   #使用ligand——MPNN，专为小分子设计，如核酸配体等等
python3 step_3_af2monomer.py -ir /home/dupuquan/RFDF3/validation_RFDF3/test -ic /home/dupuquan/RFDF3/rfd3_ppi_tutorial/output_cif -o /home/dupuquan/RFDF3/validation_RFDF3/test_output -n 14