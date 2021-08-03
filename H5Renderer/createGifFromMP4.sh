ffmpeg -i movie.mp4  -vf "fps=12,scale=1020:-1:flags=lanczos" -c:v pam -f image2pipe - | convert -delay 10 - -loop 0 -layers optimize output.gif
