from scipy.io import mmread
import numpy as np
A = mmread('./3elt_dual/3elt_dual.mtx')
b = mmread('./3elt_dual/3elt_dual_coord.mtx')

def distance_matrix_2(x, y):
    return (((x-y)**2).sum())**0.5

np.random.shuffle(b)

relation = np.array([ [distance_matrix_2(x, y) for x in b ] for y in b])
np.savetxt("3elt_dual/relation.csv", relation, delimiter=",")

size_small = 1000

relation_s = np.array([ [distance_matrix_2(x, y) for x in b[:1000] ] for y in b[:1000]])
np.savetxt(f"3elt_dual/relation_{size_small}.csv", relation_s, delimiter=",")
#np.savetxt("mnist/target.csv", train_data.targets, delimiter=",")