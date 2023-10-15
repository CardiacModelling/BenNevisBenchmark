
import nevis
import matplotlib
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tqdm
import multiprocessing
import os


def make_image(
        colorbar=False,
        starting_points=[], ending_points=[],
        markeredgewidth=1, markersize=4, alpha=0.3,
        fig=None,
        ax=None,
        meters2indices=None,
        plot_start=None,
        plot_end=None,
        image_only=True,
        **kwargs):

    if fig is None:
        fig, ax, _, meters2indices = nevis.plot(headless=True, **kwargs)
    starting_points = np.array(starting_points)
    ending_points = np.array(ending_points)
    if np.size(starting_points) != 0:
        sx, sy = meters2indices(starting_points[:, 0], starting_points[:, 1])
        if plot_start is None:
            plot_start = ax.plot(sx, sy, 'x', color='#0000ff',
                                 markeredgewidth=markeredgewidth, markersize=markersize, alpha=alpha)
        else:
            plot_start[0].set_xdata(sx)
            plot_start[0].set_ydata(sy)

    if np.size(ending_points) != 0:
        ex, ey = meters2indices(ending_points[:, 0], ending_points[:, 1])
        if plot_end is None:
            plot_end = ax.plot(ex, ey, 'x', color='#ff0000',
                               markeredgewidth=markeredgewidth, markersize=markersize, alpha=alpha)
        else:
            plot_end[0].set_xdata(ex)
            plot_end[0].set_ydata(ey)

    if plot_start is not None or plot_end is not None:
        fig.canvas.draw()

    if colorbar:
        heights = nevis.gb()
        vmin = np.min(heights)
        vmax = np.max(heights)
        def f(x): return (x - vmin) / (vmax - vmin)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'soundofmusic', [
                (0, '#4872d3'),             # Deep sea blue
                (f(-0.1), '#68b2e3'),       # Shallow sea blue
                (f(0.0), '#0f561e'),        # Dark green
                (f(10), '#1a8b33'),         # Nicer green
                (f(100), '#11aa15'),        # Glorious green
                (f(300), '#e8e374'),        # Yellow at ~1000ft
                (f(610), '#8a4121'),        # Brownish at ~2000ft
                (f(915), '#999999'),        # Grey at ~3000ft
                (1, 'white'),
            ], N=1024)
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=matplotlib.colors.Normalize(vmin, vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.081, pad=0.04)
    # Save the Matplotlib figure as an image in memory (in-memory file)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)  # Reset the file pointer to the beginning of the buffer
    image = Image.open(buf)
    image = image.convert("RGB")
    if image_only:
        return image

    return image, fig, ax, meters2indices


def combine_imgs(img1, img2, img3):
    w1, h1 = img1.size
    w2, h2 = img2.size
    w3, h3 = img3.size
    nh1 = h2 + h3
    nw1 = int(w1 * nh1 / h1)
    img1 = img1.resize((nw1, nh1))
    w1, h1 = img1.size
    # Calculate the size of the canvas
    canvas_width = w1 + w2
    canvas_height = h1

    # Create a new blank canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height),
                       (255, 255, 255))  # Use white background

    # Paste the first image onto the canvas at position (0, 0)
    canvas.paste(img1, (0, 0))

    # Paste the second image onto the canvas, adjacent to the first image
    canvas.paste(img2, (nw1, 0))  # Adjust the x-coordinate as needed
    canvas.paste(img3, (nw1, h2))
    return canvas


ben_x, ben_y = nevis.ben().grid
mac_h, mac_bdr = nevis.macdui()
mac_hg = mac_h.coords.grid


def make_combined_plot(
    starting_points=[],
    ending_points=[],
    fig1=None,
    ax1=None,
    meters2indices1=None,
    fig2=None,
    ax2=None,
    meters2indices2=None,
    fig3=None,
    ax3=None,
    meters2indices3=None,
    color_bar=True,
    image_only=True,
    save_path=None
):
    labels = {
        'Ben Nevis': [ben_x, ben_y],
        'Ben Macdui': mac_hg,
    }
    image1, fig1, ax1, meters2indices1 = make_image(
        starting_points=starting_points,
        ending_points=ending_points,
        colorbar=color_bar,
        labels=labels,
        fig=fig1,
        ax=ax1,
        meters2indices=meters2indices1,
        image_only=False
    )
    b = 8e3
    markersize, markeredgewidth, alpha = 10, 2, 0.5
    image2, fig2, ax2, meters2indices2 = make_image(
        starting_points=starting_points,
        ending_points=ending_points,
        fig=fig2,
        ax=ax2,
        meters2indices=meters2indices2,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        alpha=alpha,
        boundaries=[ben_x - b * 2.125, ben_x + b *
                    2.125, ben_y - b * 1.65, ben_y + b * 1.65],
        zoom=1,
        labels=labels,
        image_only=False,
    )

    image3, fig3, ax3, meters2indices3 = make_image(
        starting_points=starting_points,
        ending_points=ending_points,
        fig=fig3,
        ax=ax3,
        meters2indices=meters2indices3,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        alpha=alpha,
        boundaries=mac_bdr,
        zoom=1,
        labels=labels,
        image_only=False,
    )
    canvas = combine_imgs(image1, image2, image3)
    if save_path is not None:
        canvas.save(save_path)
    if image_only:
        return canvas
    return (canvas,
            fig1, ax1, meters2indices1,
            fig2, ax2, meters2indices2,
            fig3, ax3, meters2indices3)


def make_sequence(starting_points=[], ending_points=[], start_index=0, batch_size=None, downsampling=1):
    canvas, fig1, ax1, meters2indices1, \
        fig2, ax2, meters2indices2, \
        fig3, ax3, meters2indices3 = make_combined_plot(
            starting_points=starting_points[:start_index] if starting_points else [
            ],
            ending_points=ending_points[:start_index] if ending_points else [],
            image_only=False)

    if batch_size is not None:
        N = min(
            len(starting_points),
            start_index + batch_size,
        )
    else:
        N = len(starting_points)

    for i in tqdm.tqdm(range(start_index, N, downsampling)):
        canvas, fig1, ax1, meters2indices1, \
            fig2, ax2, meters2indices2, \
            fig3, ax3, meters2indices3 = make_combined_plot(
                starting_points=starting_points[i:i +
                                                downsampling] if starting_points else [],
                ending_points=ending_points[i:i +
                                            downsampling] if ending_points else [],
                fig1=fig1,
                ax1=ax1,
                meters2indices1=meters2indices1,
                fig2=fig2,
                ax2=ax2,
                meters2indices2=meters2indices2,
                fig3=fig3,
                ax3=ax3,
                meters2indices3=meters2indices3,
                color_bar=False,
                image_only=False,
                save_path=f'../result/is/{i:05d}.png',
            )


def create_video_from_images(image_directory, output_video_path, codec='XVID', fps=30.0):
    """
    Create a video from a directory of images.

    Args:
        image_directory (str): Path to the directory containing images.
        output_video_path (str): Path where the output video will be saved.
        codec (str): FourCC codec for video compression (default is 'XVID').
        fps (float): Frames per second for the video (default is 30.0).
    """
    # Get the dimensions (width and height) of the first image to determine video size
    first_image_path = os.path.join(
        image_directory, os.listdir(image_directory)[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Initialize the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    output_video = cv2.VideoWriter(
        output_video_path, fourcc, fps, (width, height))

    # Iterate through the images in the directory and add them to the video
    for filename in sorted(os.listdir(image_directory)):
        # Add more extensions if needed
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_directory, filename)
            frame = cv2.imread(image_path)
            output_video.write(frame)

    # Release the VideoWriter
    output_video.release()


def make_imgs(start_points=[], end_points=[], downsampling=1):
    N = len(start_points)
    B = 5

    pool = multiprocessing.Pool(processes=B)
    input_values = [
        (
            start_points,
            end_points,
            i * N // B,
            N // B if i != B - 1 else None,
            downsampling
        ) for i in range(B)
    ]
    # print(input_values)
    pool.starmap(make_sequence, input_values)
    pool.close()
    pool.join()


if __name__ == '__main__':
    make_combined_plot(save_path='empty.png')
    # import optuna
    # from algorithms import nelder_mead_multi
    # from framework import AlgorithmInstance
    # ins = AlgorithmInstance(nelder_mead_multi, optuna.trial.FixedTrial({}), -1)

    # ins.run_next()
    # ins.run_next()

    # res = ins.results[1]
    # print(res.to_dict())
    # make_imgs(start_points=res.points.tolist(), end_points=[], downsampling=50)

    # create_video_from_images('../result/is/', '../result/1.avi')
