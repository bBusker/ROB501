import matplotlib.pyplot as plt
from imageio import imread
from templates.histogram_eq import histogram_eq

def main():
    I = imread("../billboard/uoft_soldiers_tower_dark.png")
    J = histogram_eq(I)

    plt.imshow(I, cmap = "gray", vmin = 0, vmax = 255)
    plt.show()
    plt.imshow(J, cmap = "gray", vmin = 0, vmax = 255)
    plt.show()

if __name__ == "__main__":
    main()