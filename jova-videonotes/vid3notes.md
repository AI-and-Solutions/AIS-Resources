# 3Blue1Brown Video 3 Notes

## Recap

-Whats a neural networ, nuerons, layers, biases, and weights.

-What exactly is gradient descent and how its used to find which weights and biases minimize a certain cost function

-Find the cost by adding up all the squares of the differences between actual output vs expected output

-Find the negative gradient of cost function to see how the weights and biases need to be changed as to most efficiently decrease the final cost.

-Technically the gradient vector is a direction within thousands of dimensions

-After computing the negative gradient, the component associated with the weight essentially displays the magnitude of the extent nudging the weight would have on the cost - how sensitive it is.

## Back propagation

-Algorithm for computing the needed gradient

-In a bad network example with nueron values being completely random, when wanting to get a specific value in the output layer, the corresponding specific value nueron in the preceding layer should be nudged up while the other value nuerons nudged down

-Sizes of such nudges should be proportional to how far away each current value is from its current target value

-When nudging a nueron to the desired value, its weight, bias, and or activations from the previous layer can be either increased or decreased depending on what the desired value is

-Each of the nuerons whose biases (for example) would be changed has an individual, different influence on the total cost function; some are more influential than others

-Hebbian Theory: "Nuerons that fire together wire together"

-The biggest increases to weights and strengthing of connections happense between the most active nuerons and the desired nuerons to be most active

-If every positive bias connected with the specific neuron in the next layer got brighter, and every negative bias got dimmer, the activation of that nueron in the next layer would be stronger

-Increase weight in proportion to activation; change activation in proportion to weight

-No control over activations, only weights and biases

-Want all other neurons in the last layer to become less active; each of those other output neurons also expects its own pattern in the second to last layer

-The desired pattern for a specific output neuron is added on to all the other desired patterns of the other output neurons (all this in proportion to corresponding weights and magnitude of change for each neuron) - this is propagating backwards

-By adding up all desired patterns, you get a list of nudges that essentially tells the second to last layer how it should be manipulated

-Then recursively apply the same process to the relevant weights and biases that determine those values; in other words, like moving backwards in the network

-Same overall process for every other training example, record how each training example would like to change the weights and biases, and then average together the desired changes of weights and biases.

-Collection of averaged nudge to each weight and bias is the negative gradient of the cost function (or something in proportion to it)

## Stochastic Gradient Descent

-Problem: Computers take extremely long to add up influences of every training example of every gradient step

-Solution: Randomly shuffle training data and divide into a whole bunch of radnom "mini-batches", then compute the gradient descent step of each batch (Uncalculated Drunk man vs. Calculated Slow man)




