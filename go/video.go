package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
)

// VideoWriter pipes raw RGB frames into an ffmpeg child process.
type VideoWriter struct {
	cmd   *exec.Cmd
	stdin io.WriteCloser
}

func NewVideoWriter(path string, width, height, fps int, encoder string) (*VideoWriter, error) {
	args := []string{"-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
		"-s", fmt.Sprintf("%dx%d", width, height), "-framerate", fmt.Sprint(fps), "-i", "-",
		"-c:v", encoder, "-pix_fmt", "yuv420p"}
	if encoder == "libx264" || encoder == "libx265" {
		args = append(args, "-preset", "medium", "-crf", "18")
	} else {
		args = append(args, "-b:v", "40M")
	}
	cmd := exec.Command("ffmpeg", append(args, path)...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("could not start ffmpeg: %w", err)
	}
	return &VideoWriter{cmd, stdin}, nil
}

func (v *VideoWriter) Write(rgb []byte) error {
	if _, err := v.stdin.Write(rgb); err != nil {
		return fmt.Errorf("writing to ffmpeg failed: %w", err)
	}
	return nil
}

func (v *VideoWriter) Finish() error {
	v.stdin.Close()
	if err := v.cmd.Wait(); err != nil {
		return fmt.Errorf("ffmpeg exited with an error: %w", err)
	}
	return nil
}
