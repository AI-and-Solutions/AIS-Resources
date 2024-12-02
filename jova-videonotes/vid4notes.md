# 3Blue1Brown Video 4 Notes

## Backpropagation Calculus

## The Chain Rule in networks

-Remember the cost(of 1) is the square of the difference between the actual output value vs the expected output value.

- a(L) = sigmoid(w(L) * a(L-1) + b(L))
- z(L) = weighted sum = w(L) * a(L-1) + b(L)
- Therefore = a(L) = sigmoid (z(L))

- w(L), a(L-1), b(L)  ---> z(L) (with sigmoid) ---> (a(L)-y)^2 ---> Cost (C0)

- w(L-1), a(L-2), b(L-1) ---> z(L-1) ---> a(L-1)

-Every variable is just numbers

-dC0/dw(L) => When sliding dw(L), marginal changes in the number line of dw(L) barely change the value in the number line of dC0

-CHAIN RULE: dC0/dw(L) = dz(L)/dw(L) * da(L)/dz(L) * dC0/da(L) 

-Perform partial derivatives for all

-The final dC0/dw(L) will be the derivative for only the cost of a specific single training example

-Full cost involves averaging together all costs across many different training examples ---> dC/dw(L) = 1/n n-1summationsymbolk=0   dCk/dw(L)

-Full cost is just one component of the gradient vector, which is built up by the partial derivatives of the cost functions with respect to all the weights and biases

## Layers with additional neurons

-Simply just a few more indices to keep track of

-Instead of just a(L) and a(L-1) were now working with a0(L) & a1(L) and a0(L-1) & a1(L-1)

-ak(L-1) --> aj(L); wjk(L)

-zj(L) = wj0(L)a0(L-1) + wj1(L)a1(L-1) + wj2(L)a2(L-1) + bj(L)
-aj(L) = sigmoid(zj(L))

-UPDATED CHAIN RULE: dC0/dak(L-1) = nL-1summationj=0 dz(L)/dak(L-1) * daj(L)/dz(L) * dC0/daj(L)