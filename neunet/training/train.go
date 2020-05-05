package training

import (
	"white-lynx.fi/neunet/net"
)

//NOTE: Loss function is squared error
var (
	nSize int // number of nodes in network
)

type TrainingData interface {
	AsFloat() []float64
	Truth() bool
}

func (n *net.Net) TrainFromLoc(loc string) {
	// TODO: check all input data files contain valid data

	// wrapper that runs Train function for each sample in loc (directory path)
	// initialization should include calculations that only need to be done once per network. Global variables allow use in all functions
	nSize = n.sizeOfNet()
}

// this should forward-pass argument data, then backpropagate
func (n *net.Net) train(data TrainingData) {
	// modified version of net.Forward that keeps track of and returns each activation (in order of left to right, top to bottom)
	actHol, result := n.forwardTrain(data.AsFloat64())
	n.backprop(actHol, result, data)
}

func (n *net.Net) backprop(actHol []float64, result float64, data TrainingData) {
	// Outer layer derivation. NOTE: only 1 output node supported, cannot build multicategory NN
	var delta float64
	var wDelta []float64 = make([]float64, 0, len(n.Weights))

	// calculate outer delta, which is derivative of sigmoid * (y-t) where t is target result and y received
	if loss := result - data.Truth(); loss == 0 {
		delta = 0
	} else {
		delta = result * (1 - result) * loss
	}
	// calculate weighted delta, sum of layer delta * weight
	temp := 0
	for _, e := range n.Weights[len(n.Weights)-1][0].Weights {
		temp += e * delta
	}
	wDelta = append(wDelta, temp)

	// calc inner layer deltas
	for i := len(n.Weights) - 2; i >= 0; i-- {
		for _, e := range n.Weights[i] {
			for _, c := range e.Weights {
			}
		}
	}
}

func preluDer(v float64, aC float64) float64 {
	// derivative of prelu activation in respect to transfer function
	if v > 0 {
		// derivative of max(y) will always be 1
		return 1
	} else if v < 0 {
		// derivative of a*min(y) in respect of min(y) will always be a
		return aC
	}
	// put here what you want to return for non-derivable prelu when y=0
	//this is an arbitrary decision, perhaps 0.5?
	return 0
}

// special forward propagation that returns necessary information for backwards propagation
func (n *net) forwardTrain(input Input) ([]float64, float64) {
	d := input.AsFloat()
	// To hold activations used in backpropagation
	//TODO: calculate underlying array length instead?
	actHol := make([]float64, 0, 15)

	// using manual loop to stop before final layer, which uses different activation function, and does not need to be ran as a go routine
	for i := 0; i < len(n.Weights)-1; i++ {
		cHol := make([]chan float64, 0, len(n.Weights[i]))
		for k, _ := range n.Weights[i] {
			c := make(chan float64)
			cHol = append(cHol, c)
			// start go routine that calculates node end result
			go Transform(n.Weights[i][k], d, n.actCoef[k], c)
		}
		for l, e := range cHol {
			select {
			case t := <-e:
				d[l] = t
				actHol = append(actHol, t)
			}
		}
		// trim data (d) if necessary, NOTE: might be faster to just do it each time
		if len(cHol) != len(d) {
			d = d[:len(cHol)]
		}
	}
	res := net.OutTransform(n.Weights[len(n.Weights)-1][0], d)
	return actHol, res
}

func (n *net) sizeOfNet() int {
	var size int = 0
	for _, e := range n.Weights {
		size += len(e)
	}
	return size
}
