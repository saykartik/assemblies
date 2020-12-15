import pstats
from pstats import SortKey

print('===> CPU')
p = pstats.Stats('restats')
# p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats(SortKey.TIME).print_stats(20)


print('===> GPU')
p = pstats.Stats('restats_gpu')
# p.strip_dirs().sort_stats(-1).print_stats()
p.sort_stats(SortKey.TIME).print_stats(20)
