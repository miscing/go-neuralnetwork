package net

import (
	"math"
	"math/rand"
)

// TODO:method for persisting
// TODO:need to update initializer to take into account last activation is sigmoid instead of prelu

type Net struct {
	Weights [][]node
	// activation coeffienct regulates negative part of relu activation
	//should be same length as number of layers (first weight slice)
	actCoef []float64
}

type node struct {
	Weights []float64
	bias    float64 // bias unit
}

// NOTE: nodePerLayer should not include the final output layer
// Also layer sizes should never exceed previous layer size, will error if so (which is expected behavior, since less to more nodes shouldn't happen, same amount is okay)
func NewNet(nodePerLayer []int, inputSize int) *Net {
	//initialize weights
	// initialize actCoef to 0.25
	n := new(Net)
	// initialize layer matrix
	for i := 0; i < len(nodePerLayer); i++ {
		n.Weights = append(n.Weights, make([]node, 0))
	}
	// make first layer
	for i := 0; i < nodePerLayer[0]; i++ {
		node := initializeWeights(inputSize)
		n.Weights[0] = append(n.Weights[0], node)
	}
	// for i, e := range nodePerLayer {
	// do for number of layers (len of nodePerLayer) minus one which is output
	for i := 0; i < len(nodePerLayer)-1; i++ {
		// intialize weights here for layers after first
		for k := 0; k < nodePerLayer[i+1]; k++ {
			node := initializeWeights(nodePerLayer[i])
			n.Weights[i+1] = append(n.Weights[i+1], node)
		}
	}
	// output layer
	finalLay := initializeWeights(nodePerLayer[len(nodePerLayer)-1])
	n.Weights = append(n.Weights, []node{finalLay})
	// add all actCoef at initial value of 0.25
	for i := 0; i < len(nodePerLayer); i++ {
		n.actCoef = append(n.actCoef, 0.25)
	}
	return n
}

func initializeWeights(length int) node {
	res := make([]float64, 0, length)
	for i := 0; i < length; i++ {
		res = append(res, randomFloat())
	}
	return node{res, randomFloat()}
}

// this returns float for a prelu activation, TODO:should rewrite to also fill last layer with sigmoid activation values, but this is a minor issue
func randomFloat() float64 {
	return rand.NormFloat64() * math.Sqrt(2/(1+math.Pow(0.25, 2)))
}

func (n *Net) Forward(d []float64) float64 {
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
			}
		}
		// trim data (d) if necessary, NOTE: might be faster to just do it each time
		if len(cHol) != len(d) {
			d = d[:len(cHol)]
		}
	}
	res := OutTransform(n.Weights[len(n.Weights)-1][0], d)
	return res
}

func OutTransform(n node, d []float64) float64 {
	return Sigmoid(SumNode(n, d))
}

func Transform(n node, d []float64, aC float64, c chan<- float64) {
	c <- Prelu(SumNode(n, d), aC)
}

func SumNode(n node, d []float64) float64 {
	// TODO: consider parallelizing by sending values to own go routines?
	var sum float64 = 0
	for i, e := range n.Weights {
		sum += e * d[i]
	}
	return sum + n.bias
}

func Prelu(v float64, aC float64) float64 {
	if v > 0 {
		return v
	}
	return aC * v
}

func Sigmoid(v float64) float64 {
	return 1 / (1 + math.Exp(-v))
}
