#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "common.hpp"

extern char** environ;

// Pipes raw RGB frames into an ffmpeg child process.
VideoWriter::VideoWriter(const std::string& path, int width, int height, int fps,
                         const std::string& encoder) {
    std::vector<std::string> args = {"ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                                     "-f", "rawvideo", "-pix_fmt", "rgb24",
                                     "-s", std::to_string(width) + "x" + std::to_string(height),
                                     "-framerate", std::to_string(fps), "-i", "-",
                                     "-c:v", encoder, "-pix_fmt", "yuv420p"};
    if (encoder == "libx264" || encoder == "libx265")
        args.insert(args.end(), {"-preset", "medium", "-crf", "18"});
    else
        args.insert(args.end(), {"-b:v", "40M"});
    args.push_back(path);

    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);

    int fds[2];
    if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, fds[0], STDIN_FILENO);
    posix_spawn_file_actions_addclose(&actions, fds[1]);
    pid_t pid;
    int err = posix_spawnp(&pid, "ffmpeg", &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(fds[0]);
    if (err != 0) {
        close(fds[1]);
        throw std::runtime_error("could not start ffmpeg");
    }
    signal(SIGPIPE, SIG_IGN);  // report a dead ffmpeg as a write error instead
    fd_ = fds[1];
    pid_ = pid;
}

VideoWriter::~VideoWriter() {
    if (fd_ >= 0) close(fd_);
    if (pid_ > 0) waitpid(pid_, nullptr, 0);
}

void VideoWriter::write(const std::vector<uint8_t>& rgb) {
    const uint8_t* p = rgb.data();
    size_t left = rgb.size();
    while (left > 0) {
        ssize_t n = ::write(fd_, p, left);
        if (n < 0) throw std::runtime_error("writing to ffmpeg failed");
        p += n;
        left -= static_cast<size_t>(n);
    }
}

void VideoWriter::finish() {
    close(fd_);
    fd_ = -1;
    int status = 0;
    waitpid(pid_, &status, 0);
    pid_ = -1;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("ffmpeg exited with an error");
}
