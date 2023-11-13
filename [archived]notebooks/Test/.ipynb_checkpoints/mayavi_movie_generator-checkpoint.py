# img_movie.py
from pyface.timer.api import Timer

def animate(src, N=10):
    for j in range(N):
        for i in range(len(src.file_list)):
            src.timestep = i
            yield

if __name__ == '__main__':
    src = mayavi.engine.scenes[0].children[0]
    animator = animate(src)
    t = Timer(250, animator.next)