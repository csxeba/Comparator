import numpy as np

from csxnet.evolution import Population
from csxnet.utilities import euclidean

poplimit = 1000
survivors = 0.7
crossover = 0.1
mutation = 0.1
maxoffsprings = 3
ranges = [(0.0, 1.0)] * 5


references = """14	5.2	5.6	0.2	2288	52	62	66	0.6	11.5
10.7	3.25	2.75	2.75	2900	142.5	97.5	17.5	12.5	8.75
12.95	5.25	4.6	0	2300	160	140	15	21.25	0.2
10.6	2.5	6.5	1.75	1200	75	50	20	3	0.1
14.7	8	8	0	1450	175	80	20	4.5	0.25""".split("\n")
references = [l.split("\t") for l in references]
references = [[float(d) for d in l] for l in references]

# 3CD6	Gránátalma	15.1	0.5	13.45	0	187	27	18	254	0.8	7.2
sample = [float(d) for d in "15.1	0.5	13.45	0	187	27	18	254	0.8	7.2".split("\t")]

def phenotype(genotype):
    ph = [0] * len(genotype)
    for i, gen in enumerate(genotype):

        ph.append(gen * param)

def fitness(ind, queue=None):
    phenotype = ind.genome



Population()

