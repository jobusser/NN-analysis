# Comparison of Gradient-Based FeedForward Neural Network Training Algorithms
I tested the performance of neural network training algorithms.
The algorithms in question was stochastic gradient descent (SGD), scaled conjugate gradient (SCG) and leap frog.
At the end of the day, SGD performed the best, attesting to why it is the most widely used.

## A little more detail

#### The other algorithms
Leap frog and SCG have the attractive qualities of having guaranteed convergence (stopping at a local minimum) after a specific number of iterations.
I implemented the original algorithms of these two as propsed in the respective academic papers.

#### The tests
The data was found online and introduce problems of varying difficulty. MNIST, or handwritten digit classification, is the toughest amongst them.
The tests involved several iterations to get an overall average.

It does take a long time to run (~1 hour).

#### The results
Various improvements to leap frog and SCG, which are guaranteed to converge in a certain number of iterations unlike SGD, have probably been
improved upon since I used old academic papers for their implementations.

The file 


## Source files
* All results are in notebooks, with the graphs they produced in ``artefacts''. It may not be the most clear as to what belongs where. I will hopefully catch some time to improve it.
* data can be found in ``data''.


