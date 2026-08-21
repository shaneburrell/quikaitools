package vision

import (
	"image"
	"image/png"
	"io"
)

func encodePNG(w io.Writer, img image.Image) error {
	return png.Encode(w, img)
}
