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
acc = 100
xshift = 50
yshift = 50
zoomshift = 1.50
# -----------------


def load_foil(name):
    with open(f"configs/{name}.json") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    data = load_foil("Parafoil_4")
    foil = constructor.Parafoil(data, acc, 1.0)
    foil.export(acc, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, "Parafoil_4")
    foil_dxf = foil.airfoils[4].to_dxf(0.01, 0.01, 0.02, foil.back_offset[4], foil.top_cutoff_side[4],
                                       foil.top_cutoff[4], foil.bottom_cutoff[4], foil.rib_cutoff[4], acc)
    #foil_dxf = foil.cell_to_dxf(0, "top", acc, 0.01, 0.02, 0.01)
    #foil_dxf = foil.cell_to_dxf(0, "bot", acc, 0.02, 0.02, 0.01)
    #foil_dxf = foil.cell_to_dxf(0, "top", acc, 0.02, 0.02, 0.01, 0.01, debug=False)
    #foil_dxf = foil.cell_to_dxf(1, "bot", acc, 0.02, 0.02, 0.01)
    scale_dxf = min((simSize[0] / 2) / (foil_dxf.limits[0][1] - foil_dxf.limits[0][0]),
                    (simSize[1] / 2) / (foil_dxf.limits[0][1] - foil_dxf.limits[0][0]),
                    (simSize[0] / 2) / (foil_dxf.limits[1][1] - foil_dxf.limits[1][0]),
                    (simSize[1] / 2) / (foil_dxf.limits[1][1] - foil_dxf.limits[1][0]),
                    (simSize[0] / 2) / (foil_dxf.limits[2][1] - foil_dxf.limits[2][0]),
                    (simSize[1] / 2) / (foil_dxf.limits[2][1] - foil_dxf.limits[2][0])) * 0.9
    scale = min((simSize[0] / 2) / (foil.limits[0][1] - foil.limits[0][0]),
                (simSize[1] / 2) / (foil.limits[0][1] - foil.limits[0][0]),
                (simSize[0] / 2) / (foil.limits[1][1] - foil.limits[1][0]),
                (simSize[1] / 2) / (foil.limits[1][1] - foil.limits[1][0]),
                (simSize[0] / 2) / (foil.limits[2][1] - foil.limits[2][0]),
                (simSize[1] / 2) / (foil.limits[2][1] - foil.limits[2][0])) * 1.25
    origin = np.asarray(((simSize[0] / 4 * 3 - np.average(foil.limits[2]) * scale,
                          simSize[1] / 4 * 3 + np.average(foil.limits[1]) * scale),
                         (simSize[0] / 4 - np.average(foil.limits[0]) * scale,
                          simSize[1] / 4 + np.average(foil.limits[2]) * scale),
                         (simSize[0] / 4 - np.average(foil.limits[0]) * scale,
                          simSize[1] / 4 * 3 + np.average(foil.limits[1]) * scale),
                         (simSize[0] / 4 * 3 - np.average(foil_dxf.limits[0]) * scale_dxf,
                          simSize[1] / 4 + np.average(foil_dxf.limits[1]) * scale_dxf)))
    pg.init()
    pg.font.init()
    font = pg.font.SysFont('Arial', 20)
    pg.display.set_caption('Parafoil viewer')
    gameDisplay = pg.display.set_mode((simSize[0], simSize[1]))
    clock = pg.time.Clock()
    ended = False
    x = 0
    y = 0
    zoom = 0
    lines = len(foil_dxf.lines)

    while not ended:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                ended = True
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    x += 1
                if event.key == pg.K_RIGHT:
                    x -= 1
                if event.key == pg.K_DOWN:
                    y -= 1
                if event.key == pg.K_UP:
                    y += 1
                if event.key == pg.K_p:
                    lines = max(lines - 1, 0)
                if event.key == pg.K_n:
                    lines += 1
                if event.key == pg.K_PLUS or event.key == pg.K_KP_PLUS:
                    zoom += 1
                    x *= zoomshift
                    y *= zoomshift
                if event.key == pg.K_MINUS or event.key == pg.K_KP_MINUS:
                    zoom -= 1
                    x /= zoomshift
                    y /= zoomshift
        gameDisplay.fill(black)
        pg.draw.line(gameDisplay, white, [0, simSize[1] / 2], [simSize[0], simSize[1] / 2])
        pg.draw.line(gameDisplay, white, [simSize[0] / 2, 0], [simSize[0] / 2, simSize[1]])
        foil.draw(gameDisplay, origin[0], scale, "x", red, green, blue, np.asarray([[0.0, simSize[0]],
                                                                                    [0.0, simSize[1]]]), True)
        foil.draw(gameDisplay, origin[1], scale, "y", red, green, blue, np.asarray([[0.0, simSize[0]],
                                                                                    [0.0, simSize[1]]]), True)
        foil.draw(gameDisplay, origin[2], scale, "z", red, green, blue, np.asarray([[0.0, simSize[0]],
                                                                                    [0.0, simSize[1]]]), True)
        foil_dxf.draw(gameDisplay, (origin[3, 0] + x * xshift,
                                    origin[3, 1] + y * yshift), scale_dxf * zoomshift ** zoom, "z", red, white,
                      np.asarray([[simSize[0] / 2, simSize[0]], [0.0, simSize[1] / 2]]), lines)
        pg.draw.circle(gameDisplay, blue, (max(min(origin[3, 0] + x * xshift, simSize[0]), simSize[0] / 2),
                                           max(min(origin[3, 1] + y * yshift, simSize[1] / 2), 0.0)), 2)
        gameDisplay.blit(font.render(str(lines), False, blue), (simSize[0] / 2, 10))

        pg.display.update()
        clock.tick(60)
    pg.quit()
