import numpy as np

from csxnet.evolution import Population
from csxnet.nputils import euclidean

poplimit = 300
survivors = 0.7
crossover = 0.1
mutation = 0.1
maxoffsprings = 3
ranges = [(0.0, 1.0)] * 3

refnames = ["Gránátalma", "Barack", "Meggy"]

references = """14	5.2	5.6	0.2	2288	52	62	66	0.6	11.5
10.7	3.25	2.75	2.75	2900	142.5	97.5	17.5	12.5	8.75
12.95	5.25	4.6	0	2300	160	140	15	21.25	0.2""".split("\n")
references = [l.split("\t") for l in references]
references = np.array([[float(d) for d in l] for l in references])

# 3CD6	Gránátalma	15.1	0.5	13.45	0	187	27	18	254	0.8	7.2
sample = np.array([float(d) for d in "15.1	0.5	13.45	0	187	27	18	254	0.8	7.2".split("\t")])


def fitness(ind, queue=None):
    genotype = np.array(ind.genome)
    genotype /= genotype.sum()
    phenotype = genotype.dot(references)
    ind.fitness = euclidean(phenotype, sample)
    if queue:
        queue.put(ind)


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

myPop = Population(poplimit, survivors, crossover, mutation, fitness, maxoffsprings, ranges, parallel=True)
myPop.run(500, 2, True, True)

