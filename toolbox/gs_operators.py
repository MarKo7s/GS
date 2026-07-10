import cupy as cp


cp.fuse()
def complex_field(B, A):
    return cp.multiply(cp.absolute(B), cp.exp(1j*cp.angle(A)))

cp.fuse()
def abs2(D):
    return cp.absolute(D)**2

def power_field(D):
    return cp.sum(abs2(D))

cp.fuse()
def normalize_field(D, Dpow):
    return D / cp.sqrt(Dpow)
    
cp.fuse()
def create_field(target, Epow, scatter_field, w):
    return ((target / Epow ) * w) + scatter_field