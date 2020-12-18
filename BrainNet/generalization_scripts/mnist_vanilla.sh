# Created by Basile Van Hoorick, December 2020.

dataset_down='mnist'
n_down=784
m_down=10
data_size=10000
num_runs=5
model='ff'

for num_hidden_layers in 1 2 3
do
for hidden_width in 100 500
do
proj_cap=$(($hidden_width / 2))

# for model in table_prepost table_prepostcount table_prepostpercent table_postcount reg_oneprepost reg_oneprepostall reg_onepostall reg_allpostall

echo num_hidden_layers = $num_hidden_layers
echo hidden_width = $hidden_width
echo proj_cap = $proj_cap
echo 'Now running vanilla FF...'

# Vanilla feed-forward.
python run_eval.py --model ff --num_hidden_layers $num_hidden_layers --hidden_width $hidden_width --proj_cap $proj_cap --dataset_down $dataset_down --n_down $n_down --m_down $m_down --data_size $data_size --num_runs $num_runs --vanilla 1

# python run_eval.py --num_hidden_layers $num_hidden_layers --hidden_width $hidden_width --proj_cap $proj_cap --universal $universal --dataset_up $dataset_up --dataset_down $dataset_down --n_up $n_up --n_down $n_down --m_up $m_up --m_down $m_down --data_size $data_size --num_runs $num_runs --downstream_backprop $downstream_backprop --vanilla 0

done
done
