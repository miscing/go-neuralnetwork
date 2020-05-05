package main

import (
	"fmt"
	"math/rand"
	"white-lynx.fi/neunet/net"
)

type testData struct {
	data []float64
}

func (d testData) AsFloat() []float64 {
	return d.data
}

func genTestData(size int) testData {
	// generate test data
	hol := make([]float64, size)
	for i, _ := range hol {
		hol[i] = rand.NormFloat64()
	}
	return testData{hol}
}

func main() {
	// TODO: rand library is using the same seed, fix dis
	dataSize := 5
	for i := 0; i < 20; i++ {
		d := genTestData(dataSize)
		n := net.NewNet([]int{3, 2, 2, 2}, dataSize)
		prediction := n.Forward(d.AsFloat())
		fmt.Println(d.AsFloat(), prediction)
	}

	// for i, e := range n.Weights {
	// 	fmt.Println("layer:", i)
	// 	fmt.Println(e)
	// }
}
