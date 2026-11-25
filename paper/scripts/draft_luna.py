#!/usr/bin/env python3
"""Layout drafts of part 2 (luna-cpp showcase): luna-cpp's own figures (docs/figures, read only) tiled at 190 mm.
Raster drafts for choosing the layout only; the final versions are to be redrawn from luna-cpp's stored npz."""
import os

import matplotlib.pyplot as plt
from PIL import Image

from paperfig import DOUBLE, MM, OUT

LUNA = "/raven/u/uli/luna-cpp/docs/figures"
LAYOUTS = {  # name: rows of luna-cpp figure names
    "draft_luna_sedov": [["sedov3d_mesh"], ["sedov3d_radial", "sedov3d_shock"]],
    "draft_luna_kh": [["kh_2d_evolution"], ["kh_2d_mesh"]],
    "draft_luna_vortex": [["vortex_3d"]],
    "draft_luna_1d": [["time_to_target_1d", "super_tube_t2_luna"]],
}


def main():
    for name, rows in LAYOUTS.items():
        images = [[Image.open(os.path.join(LUNA, f + ".png")).convert("RGB") for f in row] for row in rows]
        # each row fills the width: its images share one height
        heights = [DOUBLE / sum(im.width / im.height for im in row) for row in images]
        fig = plt.figure(figsize=(DOUBLE * MM, (sum(heights) + 4 * len(rows)) * MM))
        total, y = sum(heights) + 4 * len(rows), 0.0
        for row, h in zip(images, heights):
            x = 0.0
            for im in row:
                w = h * im.width / im.height
                ax = fig.add_axes([x / DOUBLE, 1 - (y + h) / total, w / DOUBLE, h / total])
                ax.imshow(im)
                ax.axis("off")
                x += w
            y += h + 4
        fig.text(0.995, 0.003, "draft layout: luna-cpp figures as they are", ha="right", va="bottom", fontsize=6,
                 color="0.5")
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
        plt.close(fig)
        print(name, f"{DOUBLE:.0f} x {total:.0f} mm")


if __name__ == "__main__":
    main()
