from typing import Protocol
import pickle
with open('solucoes.pkl', "rb") as fh:
  data = pickle.load(fh)

with open("solucoesp4.pkl", "wb") as infile:
    pickle.dump(data, infile, protocol=4)