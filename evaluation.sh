echo "Launching evalution bash script...!"

echo "What's the repository ssh address ?"

read path_to_git

echo "What's the name of the group ?"

read group_name

git clone $path_to_git $group_name

cd $group_name

echo "Do you want to evaluate models on test (y) or not (n) ?"

read eval_type

if [ $eval_type = "n" ]
then
	path_to_data="../data"
else
	echo "Evaluating on test data"
	path_to_data="../data/test_data"
fi

echo $path_to_data

python evaluation.py $path_to_data