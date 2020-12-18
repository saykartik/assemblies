# Created by Basile Van Hoorick, December 2020.

dataset_up='halfspace'
n_up=10
m_up=2
data_size=10000
num_runs=5
hidden_width=32
proj_cap=16

for model in table_prepost table_prepostcount
do
echo model = $model

# Output rule only.
python run_eval.py --model $model --use_graph_rule 0 --use_output_rule 1 --num_hidden_layers 1 --hidden_width $hidden_width --proj_cap $proj_cap --universal 1 --dataset_up $dataset_up --n_up $n_up --m_up $m_up --data_size $data_size --num_runs $num_runs --upstream_only 1 --store_rules 1

# Hidden rule only.
python run_eval.py --model $model --use_graph_rule 1 --use_output_rule 0 --num_hidden_layers 2 --hidden_width $hidden_width --proj_cap $proj_cap --universal 1 --dataset_up $dataset_up --n_up $n_up --m_up $m_up --data_size $data_size --num_runs $num_runs --upstream_only 1 --store_rules 1

# Both rules.
python run_eval.py --model $model --use_graph_rule 1 --use_output_rule 1 --num_hidden_layers 2 --hidden_width $hidden_width --proj_cap $proj_cap --universal 1 --dataset_up $dataset_up --n_up $n_up --m_up $m_up --data_size $data_size --num_runs $num_runs --upstream_only 1 --store_rules 1

done
