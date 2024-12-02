# 3Blue1Brown Video 2 Notes

## Recap


-Nuerons have different gradients of activation, paired each with weights, added to a bias, and all put within a function; like a sigmoid.


## Using training data


-Like any other experiment, to truly see consistency in results, the experiment must be ran several times.  When training AI, after seeing results in one test case, it must be challenged with other, more difficult, test cases to improve its performance.

-Can be compared to calculus - must find minima of a certain function.

## Concept

-Each neuron in the second layer is connected with every neuron in the first layer

-Weights defining the activation are like the strengths of those connections;

-Bias is the indication of whether the nueron is active or inactive

-Weights and Biases must be initialized because if not, then the total random result will be unfavorable

-Must define a cost function: This essentially serves as the rewarding system for a computer, telling it which nuerons *should* be activated and which *shouldn't*

-What the cost means mathematically: Add up the squares of the differences between each of the "trash" output activations and the desired value. This sum will be the cost of a single training function.

-The smaller the cost, the more on point the computer is with its results. If the cost for a network is large, then the network is very inaccurate in its function.

-Take the average of the costs over the tens of thousands of training examples - this will measure and determine whether the computer should be rewarded or reprimanded

-Despite the cost function adding on another layer of complexity, it can be illustrated as a simple function with one number as input and one number as an output.

-In this function you want to find all local minima, do this by determining at a point where the slope leads to a decrease in a value - then follow that values in the descening rate until you reach the local minimum.


## Gradient Descent

-In a function with two inputs and one output, think of the input space as an (x,y) plane the cost function has as a surface.

-Incorporate multi-var concepts when finding the steepest slope of the cost function. Negative Gradient of a function will get the direction of steepest descent.

-Want to find what values in the input lead to steep downhill descent that will decrease the output value most quickly

-This idea can be used for all cost points in the network trainings

-The algorithm for computing this gradient efficiently is called ***BACK PROPAGATION***

-Biggest priority, minimizing the cost function

-Gradient Descent: Process of nudging an input of a function by some multiple of the negative gradient

-Some connections are more important than others; relative magnitudes of components dilineate which changes matter more

-An adjustment to one of the weights might have a greater impact on the cost function than the adjustment on some other weight

-Gradient vector of the cost function essentially conveys the relative importance of each weight and bias

-In other words, think of it as finding the direction of steepest descent at a point; also which of the two inputted variables has stronger importance 3/2 > 1/2

-Old techonology method - "Multilayer Perceptron"

-AI only knows a world of numbers, processes and inputs. Cannot simply just illustrate images.