# 实验记录

本地 CPU 测试自动写入本目录的日期子目录。它们均被 `.gitignore` 排除。
HPC 记录写入 `$FLOW3D_ROOT/results/experiments/`，不写入 HOME 中的代码快照。

每次运行保存有效配置、启动命令、Git 提交、数据版本、随机种子、环境、设备信息、
耗时和结果；训练运行还保存 checkpoint。具体路径和复现命令见项目 README。
论文需要的简短结果说明可人工整理为文档后提交 GitHub，原始运行产物不进入仓库。
