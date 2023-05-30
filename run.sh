python ../runKGSCL.py --dataset toys --train_batch 512 --lamda1 0.1 --lamda2 1.0 --insert_ratio 0.2 --substitute_ratio 0.7 --mark lamda1-0.1+lamda2-1.0+insert-0.2+sub-0.7
python ../runKGSCL.py --dataset grocery --train_batch 512 --lamda1 0.1 --lamda2 0.5 --insert_ratio 0.4 --substitute_ratio 0.6 --mark lamda1-0.1+lamda2-0.5+insert-0.4+sub-0.6
python ../runKGSCL.py --dataset home --train_batch 512 --lamda1 0.1 --lamda2 1.0 --insert_ratio 0.4 --substitute_ratio 0.5 --mark lamda1-0.1+lamda2-1.0+insert-0.4+sub-0.5
python ../runKGSCL.py --dataset sports --train_batch 512 --lamda1 0.1 --lamda2 0.5 --insert_ratio 0.3 --substitute_ratio 0.6 --mark lamda1-0.1+lamda2-0.5+insert-0.3+sub-0.6
