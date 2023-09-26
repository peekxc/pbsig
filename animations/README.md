
To compile the animation with decent quality, use FFMPEG: 

> ffmpeg -i frame_%03d.png -vf "fps=22,scale=800:425:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" out.gif