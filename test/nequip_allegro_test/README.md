# 使用步骤
1. 修改源文件并打成压缩包 `zip -r fasteq-NequIP-OAM-S-0.1.nequip.zip NequIP_OAM_R_4.5_l_1_NL_2_S_30_EF_SWA_nTF32_LR_1e-4_WM_0.1024-11.nequip`
2. 基于修改的源文件编译为lammps模型 `bash compile_lammps_model.sh`
3. 测试编译的lammps模型 `bash run.sh`
