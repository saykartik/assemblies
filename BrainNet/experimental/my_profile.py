# Copy whatever main into here

args = parser.parse_args()
    
cProfile.run("main(args)", 'restats_gpu')

# python my_profile.py --num_runs 1 --num_rule_epochs 10 --data_size 500 --ignore_if_exist 0 --use_gpu 1
    
    # main(args)

