#!/bin/bash

# The size of training samples batch
#K=$1;
# Number of training batches
Iterations=$1;

#Seeds=$3;


#TaskName="WDC/wdc_cameras_title_medium"
#SourceTask="WDC/wdc_shoes_title_medium"
#TaskName="Structured/Amazon-Google"
#SourceTask="Structured/Walmart-Amazon"
#TaskName="Structured/Walmart-Amazon"
#SourceTask="Structured/Walmart-Amazon"
TaskName="Structured/Beer"


InputPath="data/er_magellan/"
OutputPath="output/"

TestPath="/test.txt"

#InputPath="data/er_magellan/Structured/Beer/test.txt"
#OutputPath="output/${TaskName}/"

#InputPath="data/er_magellan/Structured/Walmart-Amazon/"
#OutputPath="output/er_magellan/Structured/Walmart-Amazon/Walmart-Amazon/"
#InputPath="data/wdc/cameras/title/"
#OutputPath="output/wdc/cameras/title/shoes/"
#InputPath="data/wdc/cameras/title/"
#OutputPath="output/wdc/cameras/title/shoes/"




LM="roberta"



declare -i MaxLen=512
declare -i Batch=32
declare -i N_Epochs=40


declare -a dataSets=(
					"Structured/Beer"
					"Dirty/DBLP-ACM"
					"Dirty/DBLP-GoogleScholar"
					"Dirty/iTunes-Amazon"
					"Dirty/Walmart-Amazon"
					"Structured/Amazon-Google"
					"Structured/Beer"
					"Structured/DBLP-ACM"
					"Structured/DBLP-GoogleScholar"
					"Structured/Fodors-Zagats"
					"Structured/iTunes-Amazon"
					"Structured/Walmart-Amazon"
					"Textual/Abt-Buy"
					)
declare -a num_epochs=(15 15 40 15 15 40 15 15 40 40 15 15)
						

# get length of an array
numOfDatasets=${#dataSets[@]}

#for Mode in ${Modes[*]}:
#do
for (( i=0; i<numOfDatasets; i++ ))
do
	echo dataSet: ${dataSets[i]}
	TaskName="${dataSets[i]}"


	Input="${InputPath}${TaskName}${TestPath}"
	
	IFS='/'     # hyphen (-) is set as delimiter
	read -ra Out <<< "$TaskName"   # str is read into an array as tokens separated by IFS
	IFS=' '

	#echo oooo: ${Out[-1]}
	output="${OutputPath}${Out[-2]}${Out[-1]}.jsonl"
	
	
	
	N_Epochs=${num_epochs[i]}
	
	
	
	for (( iter=0; iter<=Iterations; iter++ ))
	do

		echo Iteration: $iter
		# deosnt know to create folders
		#output="${OutputPath}${iter}.jsonal"
		#output="output/beer${iter}.jsonal"
		echo $output
		
		# split the string according to '/' into an array
		#IFS='/'     # hyphen (-) is set as delimiter
		#read -ra Out <<< "$TaskName"   # str is read into an array as tokens separated by IFS
		#IFS=' '

		#echo oooo: ${Out[-1]}
		#output="${OutputPath}${Out[-1]}.jsonal"
		#echo oooo: $output
		#output="${OutputPath}${&iter}.jsonal"


		CUDA_VISIBLE_DEVICES=0 python3 train_ditto.py \
				--task=${TaskName} \
				--batch=${Batch} \
				--max_len=${MaxLen} \
				--n_epochs=${N_Epochs} \
				--finetuning  \
				--lm=${LM} \
				--fp16 \
				--save_model

		CUDA_VISIBLE_DEVICES=0 python3 matcher.py \
				--task=${TaskName} \
				--input_path=${Input}  \
				--output_path=${output}  \
				--lm=${LM} \
				--use_gpu \
				--fp16 \
				--max_len=${MaxLen} \
				--checkpoint_path checkpoints/


	done
done
echo finished running
#done

#python email_sender.py \
#        --message="Run ${TaskName} Finished"