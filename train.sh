#!/bin/bash
teacher_weights="spTransformer"
student_dir="exps/upfall_fall_wokd/student/wristst_acc_distilled"
work_dir="exps/upfall_kd/student/wrist_acc_with_vanilla_kd"
student_weights="ttfstudent"
teacher_dir="$HOME/LightHART/exps/upfall_fall_wokd/teacher/skeleton_with_experimental_new_2025-07-28_21-45-38"
result_file="result.txt"
# weights="berkley_best.pt"
# work_dir="exps/bmhad_woKD/late_fusion_epoch150_alldrop0.4"

# #Utd student without KD
#python3 main.py --config ./config/utd/student.yaml --model-saved-name $student_weights --work-dir $student_dir --device 7 --base-lr 2.5e-3 --include-val True
#python3 main.py --config ./config/utd/student.yaml --work-dir $work_dir  --weights "$work_dir/$student_weights" --device 7 --base-lr 2.5e-3 --phase 'test'


#Utd teacher 
#python3 main.py --config ./config/utd/teacher.yaml --work-dir $work_dir --model-saved-name $teacher_weights  --device 7  --base-lr 2.5e-3 --phase 'train' --result-file $work_dir/$result_file  --include-val True
#python3 main.py --config ./config/utd/teacher.yaml --work-dir $work_dir  --weights "$work_dir/$teacher_weights" --device 7 --base-lr 2.5e-3 --phase 'test'

#berkley_student
#python3 main.py --config ./config/berkley/student.yaml --work-dir "$student_dir"  --model-saved-name "$student_weights" --device 7 --base-lr 2.5e-3 --include-val True
#python3 main.py --config ./config/berkley/student.yaml --work-dir "$student_dir"  --weights "$student_dir/$student_weights" --device 7 --base-lr 2.5e-3 --phase 'test'

#utd student
#python3 main.py --config ./config/czu/student.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True


#czu 
#python3 main.py --config ./config/czu/student.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True
#python3 distiller.py --config ./config/czu/distill.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True

#smartfallmm
#skelton_only experiment 
#python3 main.py --config ./config/smartfallmm/teacher.yaml --work-dir $work_dir --model-saved-name $teacher_weights  --device 1  --base-lr 2.5e-3 --phase 'train' --result-file $work_dir/$result_file  --include-val True

#multimodal experiment
#python3 main.py --config ./config/smartfallmm/teacher.yaml --work-dir $teacher_dir --model-saved-name $teacher_weights  --device 2 --base-lr 1e-3 --include-val True

################# SMARTFALLMM DISTILLATION ##############
#accelerometer only experiment
#python main.py --config ./config/smartfallmm/student.yaml --work-dir $student_dir --model-saved-name $student_weights --device 2  --include-val True
#python main.py --config ./config/smartfallmm/teacher.yaml --work-dir $teacher_dir --model-saved-name $teacher_weights --device 1 --base-lr 1e-3 --include-val True


#distillation 
#python3 distiller.py --config ./config/smartfallmm/distill.yaml --work-dir $work_dir  --teacher-weight "$teacher_dir/$teacher_weights" --model-saved-name "$student_weights" --device 2 --include-val True


############## UPFALL DISTILLATION ##########
#python3 main.py --config ./config/upfall/teacher.yaml --work-dir $teacher_dir --model-saved-name $teacher_weights  --device 2  --base-lr 1e-3 --phase 'train' --result-file $work_dir/$result_file  --include-val True
#python3 main.py --config ./config/upfall/student.yaml --work-dir $student_dir --model-saved-name $student_weights --device 2  --include-val True   
python3 distiller.py --config ./config/upfall/distill.yaml --work-dir $work_dir  --teacher-weight "$teacher_dir/$teacher_weights" --model-saved-name "$student_weights" --device 2 --include-val True