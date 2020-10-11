import matplotlib.pyplot as plt

i = (1, 0)
j = (0, 1)

I = ((1, 0), (0, 1))

# length of a vector
def vector_length(a):
    return (a[0] ** 2 + a[1] ** 2) ** (1 / 2)

# dot product between two vectors
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

# plot a list of vectors using matplotlib
def plot_vectors(list_vectors):
    for vector in list_vectors:
        plt.arrow(0, 0, vector[0], vector[1], head_width=0.2)

    plt.show()

vectors = [(0, 1), (1, 0), (2, 4), (-3, -7)]
plot_vectors(vectors)

#determinant of a matrix
def determinant(a):
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]

#product between a matrix and a vector
def matrix_vector_product(matrix, vector):
    # reuse the dot product
    return (dot(matrix[0], vector), dot(matrix[1], vector))

mat_a = ((1, 2), (3, 4))
vec_b = (5, 6)
matrix_vector_product(mat_a, vec_b)

# inverse of a matrix (if it exists)
def matmul(a, b):
    b_0 = (b[0][0], b[1][0])
    b_1 = (b[0][1], b[1][1])

    return (matrix_vector_product(a, b_0),
            matrix_vector_product(a, b_1))


def inverse(a):
    det = determinant(a)
    if det == 0:
        return False

    inv = ((a[1][1] / det, -a[0][1] / det),
           (-a[1][0] / det, a[0][0] / det))

    return inv


mat_a = ((2, 4), (5, 9))
matmul(mat_a, inverse(mat_a))

# plot 2 lists of vectors, with 2 different colors
def plot_list_vectors(list_a, list_b):
    for vector in list_a:
        plt.plot(vector[0], vector[1], 'go')

    for vector in list_b:
        plt.plot(vector[0], vector[1], 'bx')

    plt.show()


vecs_a = [(0, 1), (1, 0), (2, 4), (-3, -7)]
vecs_b = [(-1, 1), (1, 2), (1, -2), (-3, -5)]
plot_list_vectors(vecs_a, vecs_b)

# list of evenly spaced vectors:
list_vecs = []
for i in range(-5, 6):
    for j in range(-3, 4):
        list_vecs.append((i, j))

# take a matrix with determinant 0:
A = ((2, 0), (1, 0))
determinant(A)

#
# project all the vectors using A:
list_proj = []
for vec in list_vecs:
    list_proj.append(matrix_vector_product(A, vec))

plot_list_vectors(list_vecs, list_proj)

B = ((1, 4), (2, -1))
list_proj = []
for vec in list_vecs:
    list_proj.append(matrix_vector_product(B, vec))

plot_list_vectors(list_vecs, list_proj)
