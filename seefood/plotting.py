import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse

from sklearn.base import clone
from PIL import Image
import numpy as np

from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.models import HoverTool


def draw_ellipse_interactive(fig, position, covariance, alpha):
    """Draw an ellipse with a given position and covariance"""
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance), 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        fig.ellipse(x=position[0], y=position[1], width=nsig * width, height=nsig * height, angle=angle, fill_alpha=alpha, fill_color='firebrick', line_width=0.0)


def plot_interactive_gmm(model, X_embedding, total_calories, scores, img_urls, width=800, height=800):
    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@img_urls" height="224" alt="@img_urls" width="224"
                ></img>
            </div>
            <div>@calories</div>
        </div>
    """

    model_embedding = clone(model)
    model_embedding.fit(X_embedding)
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.hot(mpl.colors.Normalize()(scores))
    ]
    source = ColumnDataSource(data=dict(
        x=X_embedding[:,0],
        y=X_embedding[:,1],
        img_urls=img_urls,
        colors=colors,
        calories=total_calories,
    ))
    fig = figure(plot_width=width, plot_height=height)

    w_factor = 0.2 / model_embedding.weights_.max()
    for pos, covar, w in zip(model_embedding.means_, model_embedding.covariances_, model_embedding.weights_):
        draw_ellipse_interactive(fig, pos, covar, alpha=w * w_factor)

    scatter = fig.scatter('x', 'y', size=4, fill_color="colors", line_color=None, fill_alpha=0.5, source=source)
    scatter_hover = HoverTool(renderers=[scatter], tooltips=TOOLTIPS)
    fig.add_tools(scatter_hover)
    show(fig)

def plot_interactive_scatter(X_embedding, img_urls, values=None, labels=None, width=800, height=800):
    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@img_urls" height="224" alt="@img_urls" width="224"
                ></img>
            </div>
        </div>
    """

    if values is not None:
        colors = [
            "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b, _ in 255*mpl.cm.hot(mpl.colors.Normalize()(values))
        ]
    elif labels is not None:
        colors = labels
    else:
        raise ValueError("Either values or labels must be set")

    source = ColumnDataSource(data=dict(
        x=X_embedding[:,0],
        y=X_embedding[:,1],
        img_urls=img_urls,
        colors=colors
    ))
    fig = figure(plot_width=width, plot_height=height)
    scatter = fig.scatter('x', 'y', size=4, fill_color="colors", line_color=None, fill_alpha=0.5, source=source)
    scatter_hover = HoverTool(renderers=[scatter], tooltips=TOOLTIPS)
    fig.add_tools(scatter_hover)
    return fig


def plot_images(ids, paths, max_num=20, figsize=(15,15)):
    # read images
    ids = ids[:max_num]
    images = []
    for image_id in ids:
        with Image.open(paths[image_id]) as f:
            image = f.convert("RGB")
            images.append(image)

    # plot
    length = int(np.ceil(len(images) / 4))
    f, axarr = plt.subplots(int(length), np.min([length, 4]), figsize=figsize)
    for i, img in enumerate(images):
        if length > 1:
            x, y = int(np.floor(i / 4)), i % 4
            axarr[x, y].imshow(img)
            axarr[x, y].axis('off')
        else:
            x = int(np.floor(i / 4))
            axarr.imshow(img)
            axarr.axis('off')
