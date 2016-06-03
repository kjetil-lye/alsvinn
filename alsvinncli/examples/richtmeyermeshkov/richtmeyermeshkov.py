epsilon = 1e-1
xc = x - 0.5
yc = y - 0.5
phi = atan2(yc, xc) if abs(yc) > 0 else 0

if phi < 0:
    phi += 2*pi

if "has_random_variables" in locals() and has_random_variables:
    N = len(a1)
    normalization = sum(a1)

    if abs(normalization) < 1e-8:
        normalization = N

    perturbation = epsilon * sum([a1[n] * cos(phi+b1[n]) for n in xrange(N)]) / normalization
else:
    perturbation = 0

r = sqrt((x-0.5)**2+(y-0.5)**2)
if r < 0.1:
    p = 20
else:
    p = 1

if r < 0.25 + perturbation:
    rho = 2
else:
    rho = 1
ux = 0
uy = 0


