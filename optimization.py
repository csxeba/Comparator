import numpy as np

from csxnet.nputils import euclidean

from keras.layers.core import Dense
from keras.models import Sequential

from experiment import parse_data

chain = """MA	Gyüm	Brix	Glu	Fru	Szach	K	Ca	Mg	Na	Almasav	Citromsav
3CD6	Gránátalma	15.1	0.5	13.45	0	187	27	18	254	0.8	7.2
3CDC	Barack	16.1	6.5	7.7	0.4	381	48	24	137	2.5	0.7
3CD9	Meggy	13.7	6.3	6.3	0	56	37	12	226	0.3	4.7
R1	Gránátalma	14	5.2	5.6	0.2	2288	52	62	66	0.6	11.5
R2	Barack	10.7	3.25	2.75	2.75	2900	142.5	97.5	17.5	12.5	8.75
R3	Meggy	12.95	5.25	4.6	0	2300	160	140	15	21.25	0.2
R4	Alma	10.6	2.5	6.5	1.75	1200	75	50	20	3	0.1
R5	Szőlő	14.7	8	8	0	1450	175	80	20	4.5	0.25""".split("\n")

# 3CD6	Gránátalma	15.1	0.5	13.45	0	187	27	18	254	0.8	7.2
sample = [float(d) for d in "15.1	0.5	13.45	0	187	27	18	254	0.8	7.2".split("\t")]
sample = np.array(sample)


def fitness(genotype):
    phenotype = genotype.dot(references)
    d = euclidean(phenotype, sample)
    return d


y = np.eye(5, 5)
headers, sampleid, referenceid, samples, references = parse_data(chain)

model = Sequential()
model.add(Dense(input_dim=10, output_dim=5, activation="sigmoid"))
model.compile("rmsprop", "mse")
model.fit(references, y, show_accuracy=True)

print(model.predict(samples, verbose=1))