# BrainNet Updates for Feedforward Networks
Kartik Balasubramaniam, Brett Karopczyc, Vincent Lin, Basile Van Hoorick

---
This is an overview of the major updates and additions we've made to the original codebase made available with the paper *Learning with Plasticity Rules: Generalization and Robustness* and available at https://github.com/BrainNetwork/BrainNet.

### New Class Hierarchy:

![Class Hierarchy](PlasticityRules%20Class%20Hierarchy.png?raw=true "Title")

---
### Important Files & Directories

#####Core Feedforward Plasticity Network code
FFBrainNet.py \
FFLocalNet.py \
FFLocalPlasticityRules directory \
FFLocalLegacyClasses.py

#####Common Evaluation utilities
eval_util.py \
run_eval.py

#####Rule Comparison research
compare_util.py \
compare_halfspace.py \
compare_relu.py \
compare_mnist.py \
compare_plotting.py

#####Rule Inspection research
inspect_rules.py \
inspect_scripts directory \
inspect_plots directory

#####Generalization research
plot_generalization.py \
generalization_scripts directory \
generalization_plot directory

#####Robustness research
AdversarialExamples.py \
notebooks/FF_experiments_adversarial.ipynb

---
**Note:** The remaining files in the 'notebooks' directory are mostly obsolete versions of research that has been migrated to the files above.
