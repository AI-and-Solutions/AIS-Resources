
# <font color="#FFC19C"> Video 1: </font>

## [Link 1](https://www.youtube.com/watch?v=aircAruvnKk)

> - Neuron – Thing that holds a number.<br>
> <br>
>
> - The number inside the neuron is from 1-0 and it is called the activation.<br>
> It represents how much the handwritten number looks like a number 0-9.<br>
> <br>
>
> - In theory, different layers in the network correspond to different features of a number.<br>
> Such as curve, loop, size...<br>
> <br>
>
> - Each neuron is assigned a weight as it travels to a new neuron in a new layer.<br>
> This weight is a number<br>
> You sum all the (activation times weight) terms to highlight a region of the handwritten number<br>
> you want the neuron in the next layer to focus on.<br>
> <br>
>
> - Basically, from all the inputs, ones with low weights get dissolved out.<br>
> While ones with high weights are taken more into concideration for calculations.<br>
> <br>
>
> - So that’s cool, but weighted sums are typically large numbers<br>
> So we put them in a sigmoid function to squish 'em down to a 0-->1 range.<br>
> <br>
>
> - Ok, well mabye you don’t want <strong>0</strong> to <strong>1</strong> to be the threshold for the next neuron to activate...<br>
> <em>This could be for a few reasons I do not know...</em><br>
> So, we can add/subtract to the weight to change it.<br>
> This changes the value the incoming signal needs to activate the next neuron<br>
> We call this change to the accepted weighted sum range the neuron’s <strong>bias.</strong><br>
>
> ### TL;DR
>
> Weight = What pixel pattern is the new neuron looking for?<br>
> Bias = How strong must the incoming signal be to activate the neuron?<br>
> 
> <em>When dealing with neurons linking between layers<br>
> Weight is some formula for <strong>how important this signal is to the recieving neuron</strong><br>
> Bias is <strong>what threshold must the signal meet to be considered important to the revieving neuron</strong></em><br>
<br>

 

# <font color="#FFC19C"> Video 2: </font>

## [Link 2](https://www.youtube.com/watch?v=IHZwWFHWa-w)

> - Cost function is how a machine learns. It sees how badly it performed.<br>
> <br>
>
> - It works by adding up the squares of difference between all the output activations and what they should be in an ideal world.<br>
> In the training phase you should know what the expected output is.<br>
> <br>
>
> - Obviously, when the cost is low, that means the machine is working near-perfect.<br>
> <br>
>
> - Alright great, but how will it get better: If the cost function were graphed, you want a minimum point.<br>
> You want to take the steepest (fastest) way down to the minimum.<br>
> All this sounds like the negative gradient of the cost!<br>
> <br>
>
> - Tip: Make step sizes proportional to slope so as you get closer<br>
> and your slope levels out, you get more precise with the cost adjustments.<br>
> <br>
> 
> - Now that we know some of the underlying math, it becomes clear the real way the neural network works.<br>
> It analyzes a hodge-podge mess of individual pixels and combines them at the end.<br>
> In other words, while it can give hits for a drawn number, it cannot give correct rejections.<br>
> It also gives confident, nonsense, answers when shown a scribble.<br>
> <br>
>
> - <em>Is there a way to fix this? Yes, but let's not do that yet!</em><br>
<br>

 

# <font color="#FFC19C"> Video 3: </font>

## [Link 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

> - Backpropagation is how we compute the steepest descent:<br>
> When it comes to honing-in on the lowest cost, changing a large weight value<br>
> will have a larger effect on the cost than changing a small weight value.<br>
> <br>
>
> - Look at output layer from the neural network and how it compares to expected values.<br>
> Activations that are close to expected values should be nudged a little bit and<br>
> activations that are super far should be nudged a lot.<br>
> Nudges can be either up or down.<br>
> <br>
>
> - For each output neuron, there are three ways to modify its output:<br>
>    - Change bias,<br>
>    - Change weights,<br>
>    - Change previous layers’ activations.<br>
>    - We can do the latter by seeing how the other layers' neurons compare to<br>
>    - Expected values, and change <strong>ALL</strong> of <strong>ITS</strong> weights and biases,<br>
>    - doing so for <strong>EACH</strong> individual neuron in the layer.<br>
>    - (See adding up and down arrows at 7:55)<br>
> <br>
>
> - Changing weights of neurons with high activations also has a larger effect than<br>
> Changing the weights of neurons with low activation.<br>
> <br>
>
> - <strong>Note: It is hardest to change a neuron's activation<br>
> It is also complex to change weights<br>
> While it is the easiest to change biases</strong><br>
> <br>
> 
> - Basically, you're going layer by layer backwards up the neural network,<br>
> and seeing what the desired change for each neuron in the layer is<br>
> to then find what you need to change in the layer before <em>THAT</em> to make those changes happen<br>
> <br>
> 
> - Sum the needed nudges to neuron activations together <em>somehow...</em> and you get<br>
> the net tweaking needed for each neuron in the output layer.<br>
> That is <strong>what</strong> we’re doing, but not <strong>how</strong> we're doing it because <em>thaaat</em> takes a lot of math...<br>
> <br>
> 
> - Doing this backpropagation for many many many different images is what we call training the network.<br>
> This is done by recording how much of a step along the gradient each individual component of the neural network needs<br>
> Do this for every trial you run and then average the amount you should change <strong>each component of the network.</strong><br>
> I.e. after backpropagation for trial 1, it says change weight 12 by +8, and trial 2 says change weight 12 by -5<br>
> and so on and so on... Hence: (8+ -5+...)/(number of trials) is the <em>net change</em> you should make for that <em><strong>one weight.</em></strong><br>
> Do <strong>ALL OF THIS</strong> for every weight and/or bias and/or activation until you get accurate output activations.<br>
> <br>
>
> - Funnily enough, compiling all these changes into a column vector<br>
> gets you something proportional to the negative gradient.<br>
> <br>
>
> - In all honesty, this is quite complex. So we tend to do a much simpler, good-enough approx.<br>
> <br>
>
> - The steps are as follows:<br>
>    - Randomly shuffle output activation data for all the written numbers you analyzed.<br>
>    - Divide output layer results into sections, i.e. 100 layers per section.<br>
>    - Find the cost of each subgroup, which is a good approx of the steepest descent.<br>
>    - Take each one of those steps along the gradient sequentially.<br>
>    - This process is called Stochastic Gradient Descent.<br>
> <br>
>
> - It looks a lot like a zig-zag series of small steps down the gradient path<br>
> Which is a lot more efficient than a slow, methodical, exact path down the gradient<br>
> and it gets the job done.<br>
<br>
  


# <font color="#FFC19C"> Video 4 </font>

## [Link 4](https://www.youtube.com/watch?v=tIeHLnjs5U8)

> - The activation for a given neuron is defined by the sigmoid of <br>
> the weight between neurons times the previous neuron's activation<br>
> plus the bias of the current neuron.<br>
> <br>
>
> - Put in perspective:<br>
> an activation is defined by a weight<br>
> which is also defined by an activation<br>
> which is defined by a weight and so on...<br>
> so things get quite complex.<br>
> <br>
>
> - To get the cost: take a singular output activation,<br>
> square it, and then subtract the expected value squared.<br>
> Do this for <strong>every</strong> output in the output layer.<br>
> Average the change needed for <strong>each</strong> weight and bias in the neural network.<br>
> Then repeat everything again for a new handwritten number.<br>
> That is the mathematical approach to backpropagation; This is machine learning.<br>
> <br>
>
> - <em>It should be noted that when finding the cost:<br>
> You look one component at a time to find how much<br>
> the component needs to be tweaked. For example:</em><br>
>    - In trial 1 you run through all the weights and biases and<br>
>    - How much each should change.<br>
>    - After completing all the trials, you look exculsively at<br>
>    - weight 1 and average how much it changed over all the trials<br>
>    - then look at weight 2 and average it<br>
>    - then bias 3, bias 15...<br>
>    - <em><strong>You are averaging the change in each weight and bias independently.<br>
>    - You are integrating all of their changes together at once.<br>
>    - There is no expression for some "total cost" where some singular<br>
>    - imaginary number gets moved up or down and suddenly everything is perfect.</em></strong><br>
> <br>
>
> - There is a lot of mathematical notation, multivariable derivatives with chain rules, but<br>
> if you want to see the equations, just go to 3:55 and 4:55<br>