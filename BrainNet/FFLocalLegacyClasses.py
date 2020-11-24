"""
This file is for backward compatibility only. The classes it defines should NOT be used for future development.
"""

# Imports
from FFLocalNet import FFLocalNet
from LocalNetBase import Options, UpdateScheme

# Import plasticity rules
from FFLocalPlasticityRules.TableRule_PrePost import TableRule_PrePost
from FFLocalPlasticityRules.TableRule_PrePostCount import TableRule_PrePostCount
from FFLocalPlasticityRules.TableRule_PrePostPercent import TableRule_PrePostPercent
from FFLocalPlasticityRules.TableRule_PostCount import TableRule_PostCount
from FFLocalPlasticityRules.OneBetaANNRule_PrePost import OneBetaANNRule_PrePost
from FFLocalPlasticityRules.OneBetaANNRule_PrePostAll import OneBetaANNRule_PrePostAll
from FFLocalPlasticityRules.OneBetaANNRule_PostAll import OneBetaANNRule_PostAll
from FFLocalPlasticityRules.AllBetasANNRule_PostAll import AllBetasANNRule_PostAll

# Shim classes for backward compatibility
class FFLocalTable_PrePost(FFLocalNet):
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        rule_class = TableRule_PrePost
        hl_rule = rule_class() if options.use_graph_rule else None
        output_rule = rule_class() if options.use_output_rule else None
        super().__init__(n=n, m=m, l=l, w=w, p=p, cap=cap, hl_rules=hl_rule, output_rule=output_rule, options=options, update_scheme=update_scheme)

class FFLocalTable_PrePostCount(FFLocalNet):
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        rule_class = TableRule_PrePostCount
        hl_rule = rule_class() if options.use_graph_rule else None
        output_rule = rule_class() if options.use_output_rule else None
        super().__init__(n=n, m=m, l=l, w=w, p=p, cap=cap, hl_rules=hl_rule, output_rule=output_rule, options=options, update_scheme=update_scheme)

class FFLocalTable_PrePostPercent(FFLocalNet):
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        rule_class = TableRule_PrePostPercent
        hl_rule = rule_class() if options.use_graph_rule else None
        output_rule = rule_class() if options.use_output_rule else None
        super().__init__(n=n, m=m, l=l, w=w, p=p, cap=cap, hl_rules=hl_rule, output_rule=output_rule, options=options, update_scheme=update_scheme)

class FFLocalTable_PostCount(FFLocalNet):
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        rule_class = TableRule_PostCount
        hl_rule = rule_class() if options.use_graph_rule else None
        output_rule = rule_class() if options.use_output_rule else None
        super().__init__(n=n, m=m, l=l, w=w, p=p, cap=cap, hl_rules=hl_rule, output_rule=output_rule, options=options, update_scheme=update_scheme)

class FFLocalOneModel_PrePost(FFLocalNet):
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        rule_class = OneBetaANNRule_PrePost
        hl_rule = rule_class() if options.use_graph_rule else None
        output_rule = rule_class() if options.use_output_rule else None
        super().__init__(n=n, m=m, l=l, w=w, p=p, cap=cap, hl_rules=hl_rule, output_rule=output_rule, options=options, update_scheme=update_scheme)

class FFLocalOneModel_PrePostAll(FFLocalNet):
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        rule_class = OneBetaANNRule_PrePostAll
        hl_rule = rule_class() if options.use_graph_rule else None
        output_rule = rule_class() if options.use_output_rule else None
        super().__init__(n=n, m=m, l=l, w=w, p=p, cap=cap, hl_rules=hl_rule, output_rule=output_rule, options=options, update_scheme=update_scheme)

class FFLocalOneModel_PostAll(FFLocalNet):
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        rule_class = OneBetaANNRule_PostAll
        hl_rule = rule_class() if options.use_graph_rule else None
        output_rule = rule_class() if options.use_output_rule else None
        super().__init__(n=n, m=m, l=l, w=w, p=p, cap=cap, hl_rules=hl_rule, output_rule=output_rule, options=options, update_scheme=update_scheme)

class FFLocalAllModel_PostAll(FFLocalNet):
    def __init__(self, n, m, l, w, p, cap, options=Options(), update_scheme=UpdateScheme()):
        rule_class = AllBetasANNRule_PostAll
        hl_rule = rule_class() if options.use_graph_rule else None
        output_rule = rule_class() if options.use_output_rule else None
        super().__init__(n=n, m=m, l=l, w=w, p=p, cap=cap, hl_rules=hl_rule, output_rule=output_rule, options=options, update_scheme=update_scheme)
