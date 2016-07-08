import numpy as np

from csxnet.nputils import euclidean

chain = """MA	Gyüm	Brix	Glu	Fru	Szach	K	Ca	Mg	Na	Almasav	Citromsav
3CD6	Gránátalma	15.1	0.5	13.45	0	187	27	18	254	0.8	7.2
3CDC	Barack	16.1	6.5	7.7	0.4	381	48	24	137	2.5	0.7
3CD9	Meggy	13.7	6.3	6.3	0	56	37	12	226	0.3	4.7
R1	Gránátalma	14	5.2	5.6	0.2	2288	52	62	66	0.6	11.5
R2	Barack	10.7	3.25	2.75	2.75	2900	142.5	97.5	17.5	12.5	8.75
R3	Meggy	12.95	5.25	4.6	0	2300	160	140	15	21.25	0.2
R4	Alma	10.6	2.5	6.5	1.75	1200	75	50	20	3	0.1
R5	Szőlő	14.7	8	8	0	1450	175	80	20	4.5	0.25""".split("\n")


class Sample:
    def __init__(self, name: str, data: np.ndarray, headers: np.ndarray=None):
        self.dims = data.shape[0]
        self.name = name
        self.data = data
        self.headers = headers

    def compare(self, other):
        assert self.dims == other.dims, "Dimensions differ! Cannot compare samples!"
        return euclidean(self.data, other.data)

    def mixture(self, other):
        assert self.dims == other.dims, "Dimensions differ! Can't create theoretical mixture!"
        coefs = np.arange(0, 101)
        left = np.outer(coefs, self.data)
        right = np.outer(coefs[::-1], other.data)
        mix = np.add(left, right) / 100
        return [Sample(name="{} {} + {} {}".format(i, self.name, 100-i, other.name),
                       data=mix[i], headers=self.headers) for i in coefs]

    def show(self):
        print("\n", self.name + ":")
        for h, d in zip(self.headers, self.data):
            print("{} = {}".format(h, d))

    def __repr__(self):
        return str(self.name) + ": " + str(self.data)


def test_against_mixture(sample: Sample, ref1: Sample, ref2: Sample):
    mix = ref1.mixture(ref2)
    dissims = []
    for tmixture in mix:
        dissims.append(sample.compare(tmixture))
    bestmatch = np.argmin(dissims)
    print("Sample is most similar to", mix[bestmatch].name)
    print("with dissimilarity:", dissims[bestmatch])
    print("Dissimilarity with {}: {}".format(mix[0].name, dissims[0]))
    print("Dissimilarity with {}: {}".format(mix[-1].name, dissims[-1]))
    print(sample.show())
    print(mix[bestmatch].show())


def parse_data(data):
    headers = data[0].split("\t")[2:]
    data = np.array([d.split("\t") for d in data[1:]])
    cases = data[..., :2]
    data = data[..., 2:].astype(float)

    means = data.mean(axis=0)
    stds = data.std(axis=0)

    stddata = np.copy(data)
    stddata -= means
    stddata /= stds

    samples, references = stddata[:3], stddata[3:]
    sampleid, referenceid = cases[:3], cases[3:]

    return headers, sampleid, referenceid, samples, references


def run():
    header, sampleid, referenceid, samples, references = parse_data(chain)
    print("Determining dissimilarities of samples from references...")
    for sid, sample in zip(sampleid, samples):
        for rid, reference in zip(referenceid, references):
            print("{}\t{}\t{}".format(sid[0], rid[1], str(euclidean(sample, reference)).replace(".", ",")))

    print("\nDetermining possible mixture contents of 3CD6 (Gránátalma-Alma)")
    test_against_mixture(Sample("3CD6", samples[0], header),
                         Sample("Gránátalma", references[0], header),
                         Sample("Alma", references[3], header))


if __name__ == '__main__':
    run()
