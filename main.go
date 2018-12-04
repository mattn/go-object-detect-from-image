package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/build"
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

type jsonResult struct {
	Name        string  `json:"name"`
	Probability float64 `json:"probability"`
}

func drawString(img *image.RGBA, p image.Point, c color.Color, s string) {
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c),
		Face: basicfont.Face7x13,
		Dot:  fixed.Point26_6{fixed.Int26_6(p.X * 64), fixed.Int26_6(p.Y * 64)},
	}
	d.DrawString(s)
}

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

func detectObjects(session *tf.Session, graph *tf.Graph, input *tf.Tensor) ([]float32, []float32, [][]float32, error) {
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
	var jsoninfo bool
	var probability float64
	var dir string
	var output string

	flag.BoolVar(&jsoninfo, "json", false, "Output JSON information (instead of output image)")
	flag.Float64Var(&probability, "prob", 0.4, "Probability")
	flag.StringVar(&dir, "dir", filepath.Join(filepath.SplitList(build.Default.GOPATH)[0], "src/github.com/mattn/go-object-detect-from-image"), "Directory containing the trained model and labels files")
	flag.StringVar(&output, "output", "output.jpg", "Output file name")
	flag.Parse()

	model, err := ioutil.ReadFile(filepath.Join(dir, "frozen_inference_graph.pb"))
	if err != nil {
		log.Fatal(err)
	}

	labels, err := loadLabels(filepath.Join(dir, "coco_labels.txt"))
	if err != nil {
		log.Fatal(err)
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		log.Fatal(err)
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	var f *os.File
	if flag.NArg() == 1 {
		f, err = os.Open(flag.Arg(0))
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
	} else {
		f = os.Stdin
	}
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

	probabilities, classes, boxes, err := detectObjects(session, graph, tensor)
	if err != nil {
		log.Fatalf("error making prediction: %v", err)
	}

	if jsoninfo {
		var result []jsonResult
		i := 0
		for float64(probabilities[i]) > probability {
			idx := int(classes[i])
			result = append(result, jsonResult{
				Name:        labels[idx],
				Probability: float64(probabilities[i]),
			})
			i++
		}
		json.NewEncoder(os.Stdout).Encode(result)
		return
	}

	bounds := img.Bounds()
	canvas := image.NewRGBA(bounds)
	draw.Draw(canvas, bounds, img, image.Pt(0, 0), draw.Src)
	i := 0
	for float64(probabilities[i]) > probability {
		idx := int(classes[i])
		y1 := int(float64(bounds.Min.Y) + float64(bounds.Dy())*float64(boxes[i][0]))
		x1 := int(float64(bounds.Min.X) + float64(bounds.Dx())*float64(boxes[i][1]))
		y2 := int(float64(bounds.Min.Y) + float64(bounds.Dy())*float64(boxes[i][2]))
		x2 := int(float64(bounds.Min.X) + float64(bounds.Dx())*float64(boxes[i][3]))
		drawRect(canvas, image.Rect(x1, y1, x2, y2), color.RGBA{255, 0, 0, 0})
		drawString(
			canvas,
			image.Pt(x1, y1),
			colornames.Map[colornames.Names[idx]],
			fmt.Sprintf("%s (%3.0f%%)", labels[idx], probabilities[i]*100.0))
		i++
	}

	out, err := os.Create(output)
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()

	err = jpeg.Encode(out, canvas, nil)
	if err != nil {
		log.Fatal(err)
	}
}
