import numpy as np

from csxnet.nputils import euclidean


refnames = ["Gránátalma", "Barack", "Meggy"]

references = """14	5.2	5.6	0.2	2288	52	62	66	0.6	11.5
10.7	3.25	2.75	2.75	2900	142.5	97.5	17.5	12.5	8.75
12.95	5.25	4.6	0	2300	160	140	15	21.25	0.2""".split("\n")
references = [l.split("\t") for l in references]
references = [[float(d) for d in l] for l in references]
references = np.array(references)

# 3CD6	Gránátalma	15.1	0.5	13.45	0	187	27	18	254	0.8	7.2
sample = [float(d) for d in "15.1	0.5	13.45	0	187	27	18	254	0.8	7.2".split("\t")]
sample = np.array(sample)


def fitness(genotype):
    phenotype = [0] * len(sample)
    for gen, ref in zip(genotype, references):
        for i, rparam in enumerate(ref):
            phenotype[i] += gen * rparam
    return euclidean(phenotype, sample)
