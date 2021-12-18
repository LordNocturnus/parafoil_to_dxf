import numpy as np
import pygame as pg
import json
import constructor

# --- constants ---
simSize = (1920, 1080)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)
acc = 250
# -----------------


def load_foil(name):
    with open(f"configs/{name}.json") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    data = load_foil("Parafoil_0")
    foil = constructor.Parafoil(data, acc)
    foil.export(acc, 0.02, 0.02, 0.01, 0.02, 0.02, "Parafoil_0")
    #foil_dxf = foil.airfoils[1].to_dxf(0.02, 0.02)
    #foil_dxf = foil.cell_to_dxf(0, "top", acc, 0.02, 0.02, 0.01)
    foil_dxf = foil.cell_to_dxf(0, "bot", acc, 0.02, 0.02, 0.01)
    scale = min((simSize[0] / 2) / (foil.limits[0][1] - foil.limits[0][0]),
                (simSize[1] / 2) / (foil.limits[0][1] - foil.limits[0][0]),
                (simSize[0] / 2) / (foil.limits[1][1] - foil.limits[1][0]),
                (simSize[1] / 2) / (foil.limits[1][1] - foil.limits[1][0]),
                (simSize[0] / 2) / (foil.limits[2][1] - foil.limits[2][0]),
                (simSize[1] / 2) / (foil.limits[2][1] - foil.limits[2][0])) * 0.9
    scale_dxf = min((simSize[0] / 2) / (foil_dxf.limits[0][1] - foil_dxf.limits[0][0]),
                    (simSize[1] / 2) / (foil_dxf.limits[0][1] - foil_dxf.limits[0][0]),
                    (simSize[0] / 2) / (foil_dxf.limits[1][1] - foil_dxf.limits[1][0]),
                    (simSize[1] / 2) / (foil_dxf.limits[1][1] - foil_dxf.limits[1][0]),
                    (simSize[0] / 2) / (foil_dxf.limits[2][1] - foil_dxf.limits[2][0]),
                    (simSize[1] / 2) / (foil_dxf.limits[2][1] - foil_dxf.limits[2][0])) * 0.9
    origin = np.asarray(((simSize[0] / 4 * 3 - np.average(foil.limits[2]) * scale,
                          simSize[1] / 4 * 3 + np.average(foil.limits[1]) * scale),
                         (simSize[0] / 4 - np.average(foil.limits[0]) * scale,
                          simSize[1] / 4 + np.average(foil.limits[2]) * scale),
                         (simSize[0] / 4 - np.average(foil.limits[0]) * scale,
                          simSize[1] / 4 * 3 + np.average(foil.limits[1]) * scale),
                         (simSize[0] / 4 * 3 - np.average(foil_dxf.limits[0]) * scale_dxf,
                          simSize[1] / 4 + np.average(foil_dxf.limits[1]) * scale_dxf)))
    pg.init()
    pg.display.set_caption('Parafoil viewer')
    gameDisplay = pg.display.set_mode((simSize[0], simSize[1]))
    clock = pg.time.Clock()
    ended = False

    while not ended:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                ended = True
        gameDisplay.fill(black)
        pg.draw.line(gameDisplay, white, [0, simSize[1] / 2], [simSize[0], simSize[1] / 2])
        pg.draw.line(gameDisplay, white, [simSize[0] / 2, 0], [simSize[0] / 2, simSize[1]])
        foil.draw(gameDisplay, origin[0], scale, "x", red, green, blue, white)  # , False)
        foil.draw(gameDisplay, origin[1], scale, "y", red, green, blue, white)  # , False)
        foil.draw(gameDisplay, origin[2], scale, "z", red, green, blue, white)  # , False)
        foil_dxf.draw(gameDisplay, origin[3], scale_dxf, "z", red)
        #pg.draw.circle(gameDisplay, white, origin[3], 5)

        pg.display.update()
        clock.tick(60)
    pg.quit()
