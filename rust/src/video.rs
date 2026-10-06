use std::io::Write;
use std::process::{Child, ChildStdin, Command, Stdio};

use anyhow::Context;

/// Pipes raw RGB frames into an ffmpeg child process.
pub struct VideoWriter {
    child: Child,
    stdin: ChildStdin,
}

impl VideoWriter {
    pub fn new(path: &str, width: i32, height: i32, fps: i32, encoder: &str) -> anyhow::Result<Self> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}"), "-framerate", &fps.to_string(), "-i", "-"])
            .args(["-c:v", encoder, "-pix_fmt", "yuv420p"]);
        if encoder == "libx264" || encoder == "libx265" {
            cmd.args(["-preset", "medium", "-crf", "18"]);
        } else {
            cmd.args(["-b:v", "40M"]);
        }
        let mut child = cmd.arg(path).stdin(Stdio::piped()).spawn().context("could not start ffmpeg")?;
        let stdin = child.stdin.take().unwrap();
        Ok(Self { child, stdin })
    }

    pub fn write(&mut self, rgb: &[u8]) -> anyhow::Result<()> {
        self.stdin.write_all(rgb).context("writing to ffmpeg failed")
    }

    pub fn finish(mut self) -> anyhow::Result<()> {
        drop(self.stdin);
        let status = self.child.wait()?;
        anyhow::ensure!(status.success(), "ffmpeg exited with an error");
        Ok(())
    }
}
