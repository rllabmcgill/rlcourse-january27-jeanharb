import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

#different gammas to compare
gammas = np.array([ 0.99, 0.95, 0.9, 0.8, 0.5, 0.3, 0.1])

#use random transition and reward matrix or create manually
random_P = True
if random_P:
  P_size = 15
  pi = np.random.rand(P_size,P_size)
  pi /= pi.sum(axis=1,keepdims=True)
  P = np.random.rand(P_size,P_size,P_size)
  P /= P.sum(axis=2, keepdims=True)
  current_P = (P*np.expand_dims(pi, axis=2)).sum(axis=2)
  r = np.random.rand(P_size,)
else:
  P = np.array([[0.2,0.1,0.7],[0.2,0.7,0.1],[0.7,0.2,0.1]])
  r = np.array([0.2,0.3,0.5])

#Rescale rewards such that initial norm of errors of different gammas are equivalent.
#Allows for easier comparison.
r_scales = (1-gammas)/(1-gammas.max())
r = np.expand_dims(r, axis=0)*np.expand_dims(r_scales,axis=1)

#v = gamma*P*v + r
#v-gamma*P*v = r
#v = (I-gamma*P)^(-1)*r

optimal_V = np.zeros(((len(gammas), P.shape[0])))
for i in range(len(gammas)):
  optimal_V[i] = linalg.inv(np.eye(current_P.shape[0])-gammas[i]*current_P).dot(r[i])

def err(a,b):
  return np.square(a-b).sum()

V = np.zeros((len(gammas), P.shape[0]))

print "testing the following gammas:", gammas

num_iters = 50
errs = np.zeros((num_iters, len(gammas)))

# could create pi improvement loop.
# If outside of num_iters loop, this would be policy iteration
# If inside num_iters_loop, this would be value iteration
# If number of iterations isn't enough for value to converge, then this would be modified policy iteration

# We aren't improving the policy in this step, so we set the number to 1
policy_improvement_steps = 1

for k in range(policy_improvement_steps):
  current_pi = pi # usually this would be policy improvement steps

  current_P = (P*np.expand_dims(current_pi, axis=2)).sum(axis=2)
  for i in range(num_iters):
    for j in range(len(gammas)):
      errs[i,j] = err(V[j],optimal_V[j])
      print "%.6f\t" % (errs[i,j]),
      V[j] = (gammas[j]*current_P).dot(V[j]) + r[j]
    print

plt.plot(errs)
plt.legend(gammas)
plt.show()
