import ffmpeg

(
    ffmpeg
    .input('../resources/imgs/bj/img%d.jpg', framerate=40)
    .output('movie.mp4')
    .run()
)
