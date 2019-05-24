import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


import fig_config as CONFIG

CONFIG.plot_setup()
np.random.seed(0)


def get_palette_colour(label):

    palette = CONFIG.base_palette(n=7)
    mapping = {
        'Shape': palette[0],
        'Clinical': palette[1],
        'CT First Order': palette[2],
        'PET First Order': palette[3],
        'CT Texture': palette[4],
        'PET Texture': palette[5],
        'PET parameter': palette[6]
    }
    return mapping[label]


def gen_piechart():

    path_to_figure = 'absolute_piechart.pdf'

    show = True

    labels = [
        'Shape', 'CT First Order', 'CT Texture', 'PET First Order',
        'PET Texture', 'Clinical'
    ]
    # Absolute sizes.
    sizes = [2, 6, 11, 1, 2, 4]
    # Sizes relative to feature category size.
    #sizes = [2 / 8, 6 / 12, 11 / 56, 1 / 7, 2 / 27, 4 / 42]

    # PET FS = 7
    # PET TEXT = 27
    # CT FS = 12
    # CT TEXT = 56
    # Shape = 8
    # Clinical = 42

    colors = []
    handles = []
    for key in labels:
        color = get_palette_colour(key)
        colors.append(color)
        handles.append(mpatches.Patch(color=color, label=key))

    palette = CONFIG.base_palette(n=len(sizes))
    plt.pie(
        sizes,
        colors=colors,
        autopct='%1.1f%%',
        shadow=False, startangle=0,
        labeldistance=0.85, textprops={'fontsize': 22},
        pctdistance=0.8
    )
    plt.legend(
        handles=handles,
        title='Feature Categories:',
        title_fontsize=18,
        fancybox=True,
        shadow=True,
        ncol=1,
        labelspacing=0.25,
    )
    plt.axis('equal')
    plt.savefig(
        path_to_figure, bbox_inches='tight', transparent=True, dpi=CONFIG.DPI
    )
    if show:
        plt.show()


def gen_doughnut():

    show = False
    path_to_figure = './doughnut.pdf'

    n = 5
    sizes = np.ones(n) / n
    palette = CONFIG.base_palette(n=len(sizes))
    circle = plt.Circle((0, 0), 0.7, color='white')

    plt.pie(sizes, colors=palette)
    axis = plt.gcf()
    axis.gca().add_artist(circle)
    plt.savefig(path_to_figure, bbox_inches='tight', transparent=True, dpi=CONFIG.DPI)

    if show:
        plt.show()


def gen_histogram():

    show = False
    path_to_figure = './histogram.pdf'

    palette = CONFIG.base_palette(n=1)

    data = np.random.normal(size=10000)

    plt.hist(data, bins=200, color=palette)
    plt.axis('off')
    plt.savefig(path_to_figure, bbox_inches='tight', transparent=True, dpi=CONFIG.DPI)

    if show:
        plt.show()


def gen_textures():

    show = True
    nrows, ncols = 5, 5
    path_to_figure = './texture1.pdf'

    matrix = np.random.choice([_ for _ in range(0, 5, 1)], nrows * ncols)
    matrix.resize(nrows, ncols)
    matrix = np.sort(matrix, axis=1)

    sns.heatmap(
        matrix, cmap='viridis', cbar=False, annot=True, annot_kws={'size': 60},
        linewidths=3, linecolor='k', square=True
    )
    plt.axis('off')
    plt.savefig(path_to_figure, bbox_inches='tight', transparent=True, dpi=CONFIG.DPI)

    if show:
        plt.show()


if __name__ == '__main__':
    gen_piechart()
