from tkinter import *
from experiment import Sample

import numpy as np


class App(Tk):
    def __init__(self):
        Tk.__init__(self)

        self.title("Elméleti keverék számítás!")
        Label(self, text="Elméleti keverék számítás!", font=(15,)).pack()
        self.namesframe = self.Names_Frame(self)
        self.namesframe.pack()
        self.midframe = self.Sample_Frame(self)
        self.midframe.pack()

        Button(self, text="Calculate!", command=self.calculate).pack()

    def calculate(self):
        self.midframe.refresh()
        self.namesframe.refresh()
        samples = []
        for dname in self.namesframe.defaults:
            name = self.namesframe.names[dname]
            samples.append(Sample(name, self.midframe.data[dname]))
        sample, ref1, ref2 = samples
        mix = ref1.mixture(ref2)
        dissims = []
        for theoretical_mixture in mix:
            dissims.append(sample.compare(theoretical_mixture))
        bestmatch = np.argmin(dissims)
        self.Results_Toplevel(self, bestmatch, sample, dissims, mix)

    class Names_Frame(Frame):
        def __init__(self, master):
            Frame.__init__(self, master)
            self.defaults = ("Minta", "Referencia 1", "Referencia 2")
            self.entries = {}
            self.names = {}
            for i, name in enumerate(self.defaults):
                self.names[name] = None
                Label(self, text=name, width=10, anchor=W).grid(row=i, column=0)
                self.entries[name] = Entry(self, width=22)
                self.entries[name].grid(row=i, column=1)

        def refresh(self):
            for dname in self.defaults:
                n = self.entries[dname].get()
                self.names[dname] = n if n else dname

    class Sample_Frame(Frame):
        def __init__(self, master, **kwargs):
            Frame.__init__(self, master, **kwargs)
            self.names = ("Minta", "Referencia 1", "Referencia 2")
            self.entries = {name: [] for name in self.names}
            self.botframes = {}
            self.plusminus = {name: [None, None] for name in self.names}
            self.data = {name: [] for name in self.names}

            for c in self.names:
                Label(self, text="Add meg a {} paramétereit".format(c)).pack(fill=X)

                self.botframes[c] = Frame(self, bd=3, relief=RAISED, padx=5)
                self.botframes[c].pack()
                for _ in range(3):
                    self.entries[c].append(
                        Entry(self.botframes[c], width=7)
                    )
                    self.entries[c][-1].pack(side=LEFT)

            self.add_plusminus()

        def add_plusminus(self):
            w = 3
            for name in self.names:
                mb = Button(self.botframes[name], text="-1", width=w, command=self.minus_one)
                pb = Button(self.botframes[name], text="+1", width=w, command=self.plus_one)
                self.plusminus[name][0] = mb
                self.plusminus[name][1] = pb
                mb.pack(side=LEFT)
                pb.pack(side=LEFT)

        def delete_plusminus(self):
            for t in self.plusminus.values():
                t[0].destroy()
                t[1].destroy()

        def minus_one(self):
            if len(self.entries["Minta"]) == 3:
                return

            self.delete_plusminus()
            for name in self.names:
                self.entries[name][-1].destroy()
                self.entries[name] = self.entries[name][:-1]
            self.add_plusminus()

        def plus_one(self):
            if len(self.entries) == 20:
                return

            self.delete_plusminus()
            for name in self.names:
                e = Entry(self.botframes[name], width=7)
                self.entries[name].append(e)
                e.pack(side=LEFT)
            self.add_plusminus()

        def refresh(self):
            for name in self.names:
                self.data[name] = np.array([float(e.get().replace(",", ".")) for e in self.entries[name]])

    class Results_Toplevel(Toplevel):
        def __init__(self, master, bestmatch, sample, dissims, mixture):
            Toplevel.__init__(self, master)

            Label(self, text="Számítás eredménye!")
            self.t = Text(self, width=100, height=50)
            self.t.pack()

            self.add("{} hasonlítása elméleti keverékekhez!".format(sample.name))
            self.add("Legjobb egyezés: {}. Disszimiliratás: {}".format(mixture[bestmatch].name,
                                                                       dissims[bestmatch]))
            self.add("Ehhez tartozó paramétersor: {}\n".format(str(mixture[bestmatch].data)))
            self.add("----------------------------------------------")
            self.add("Minta paraméterei: {}".format(str(sample.data)))
            self.add("Keverék elemei és a disszimilaritás:")
            for i, s in enumerate(mixture):
                self.add("{}:\t{}".format(s.name, dissims[i]))

        def add(self, tx):
            self.t.insert(END, tx + "\n")

if __name__ == '__main__':
    App().mainloop()
