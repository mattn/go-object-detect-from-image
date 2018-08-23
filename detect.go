package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"golang.org/x/image/bmp"
	"golang.org/x/image/colornames"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

var (
	session *tf.Session
	graph   *tf.Graph
)

func drawRect(img *image.RGBA, r image.Rectangle, c color.Color) {
	for x := r.Min.X; x <= r.Max.X; x++ {
		img.Set(x, r.Min.Y, c)
		img.Set(x, r.Max.Y, c)
	}
	for y := r.Min.Y; y <= r.Max.Y; y++ {
		img.Set(r.Min.X, y, c)
		img.Set(r.Max.X, y, c)
	}
}

func loadLabels(name string) ([]string, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return labels, nil
}

func decodeBitmapGraph() (*tf.Graph, tf.Output, tf.Output, error) {
	s := op.NewScope()
	input := op.Placeholder(s, tf.String)
	output := op.ExpandDims(
		s,
		op.DecodeBmp(s, input, op.DecodeBmpChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err := s.Finalize()
	return graph, input, output, err
}

func makeTensorFromImage(img []byte) (*tf.Tensor, image.Image, error) {
	tensor, err := tf.NewTensor(string(img))
	if err != nil {
		return nil, nil, err
	}
	normalizeGraph, input, output, err := decodeBitmapGraph()
	if err != nil {
		return nil, nil, err
	}
	normalizeSession, err := tf.NewSession(normalizeGraph, nil)
	if err != nil {
		return nil, nil, err
	}
	defer normalizeSession.Close()
	normalized, err := normalizeSession.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, nil, err
	}

	r := bytes.NewReader(img)
	i, _, err := image.Decode(r)
	if err != nil {
		return nil, nil, err
	}
	return normalized[0], i, nil
}

func detectObjects(graph *tf.Graph, input *tf.Tensor) ([]float32, []float32, [][]float32, error) {
	inputop := graph.Operation("image_tensor")
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputop.Output(0): input,
		},
		[]tf.Output{
			graph.Operation("detection_boxes").Output(0),
			graph.Operation("detection_scores").Output(0),
			graph.Operation("detection_classes").Output(0),
			graph.Operation("num_detections").Output(0),
		},
		nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("Error running session: %v", err)
	}
	probabilities := output[1].Value().([][]float32)[0]
	classes := output[2].Value().([][]float32)[0]
	boxes := output[0].Value().([][][]float32)[0]
	return probabilities, classes, boxes, nil
}

func main() {
	dir := flag.String("dir", "frozen_inference_graph.pb", "Directory containing the trained model and labels files.")
	flag.Parse()
	if *dir == "" {
		flag.Usage()
		return
	}
	model, err := ioutil.ReadFile(filepath.Join(*dir, "frozen_inference_graph.pb"))
	if err != nil {
		log.Fatal(err)
	}

	labels, err := loadLabels(filepath.Join(*dir, "coco_labels.txt"))
	if err != nil {
		log.Fatal(err)
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	session, err = tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	f, err := os.Open(flag.Arg(0))
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	var buf bytes.Buffer
	err = bmp.Encode(&buf, img)
	if err != nil {
		log.Fatal(err)
	}

	tensor, img, err := makeTensorFromImage(buf.Bytes())
	if err != nil {
		log.Fatalf("error making input tensor: %v", err)
	}

	probabilities, classes, boxes, err := detectObjects(graph, tensor)
	if err != nil {
		log.Fatalf("error making prediction: %v", err)
	}

	bounds := img.Bounds()
	canvas := image.NewRGBA(bounds)
	draw.Draw(canvas, bounds, img, image.Pt(0, 0), draw.Src)
	i := 0
	for probabilities[i] > 0.4 {
		idx := int(classes[i])
		y1 := int(float64(bounds.Min.Y) + float64(bounds.Dy())*float64(boxes[i][0]))
		x1 := int(float64(bounds.Min.X) + float64(bounds.Dx())*float64(boxes[i][1]))
		y2 := int(float64(bounds.Min.Y) + float64(bounds.Dy())*float64(boxes[i][2]))
		x2 := int(float64(bounds.Min.X) + float64(bounds.Dx())*float64(boxes[i][3]))
		drawRect(canvas, image.Rect(x1, y1, x2, y2), color.RGBA{255, 0, 0, 0})
		c := colornames.Map[colornames.Names[idx]]
		point := fixed.Point26_6{fixed.Int26_6(x1 * 64), fixed.Int26_6(y1 * 64)}
		d := &font.Drawer{
			Dst:  canvas,
			Src:  image.NewUniform(c),
			Face: basicfont.Face7x13,
			Dot:  point,
		}
		d.DrawString(fmt.Sprintf("%s (%2.0f%%)", labels[idx], probabilities[idx]*100.0))
		i++
	}

	out, err := os.Create("output.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()

	err = jpeg.Encode(out, canvas, nil)
	if err != nil {
		log.Fatal(err)
	}
}
